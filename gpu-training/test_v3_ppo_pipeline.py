from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_COUNT,
    V3_ACTION_FEATURE_COUNT,
    V3_ACTION_FEATURE_LAYOUT,
    V3_ACTION_FEATURES,
    V3ActionConditionedActorCriticNetwork,
    export_v3_action_conditioned_json,
)
from v3_ppo_dataset import (
    V3_PPO_SEMANTICS_CONTRACT,
    V3_PPO_SEMANTICS_CONTRACT_SHA256,
    build_v3_ppo_data_verification,
    load_v3_ppo_rollouts,
)
from run_gpu_v3_ppo import (
    EXPECTED_DETERMINISM,
    EXPECTED_PATH_POLICY,
    assert_protected_inputs_unchanged,
    expected_algorithm,
    protected_input_snapshot,
    resolve_fresh_run_paths,
    resolve_protected_inputs,
    run_after_sealing_log,
    strict_python_environment,
    validate_run_config,
)


ROOT = Path(__file__).resolve().parent


def mask_hex(indices: list[int]) -> str:
    values = [0] * (V3_ACTION_COUNT // 4)
    for index in indices:
        values[index // 4] |= 1 << (index % 4)
    return "".join(format(value, "x") for value in values)


ROLES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)


def role_for_index(index: int, player_count: int) -> str:
    if index == 0:
        return "great-dalmuti"
    if index == 1:
        return "lesser-dalmuti"
    if index == player_count - 2:
        return "lesser-peon"
    if index == player_count - 1:
        return "great-peon"
    return "merchant"


def observation(
    rank_counts: dict[int, int],
    *,
    player_count: int = 4,
    round_number: int = 1,
    actor_seat: int = 0,
) -> list[float]:
    values = [0.0] * 172
    values[0] = (player_count - 4) / 6
    values[1] = min(round_number / 20, 1)
    values[2] = actor_seat / (player_count - 1)
    actor_role = role_for_index(actor_seat, player_count)
    values[3 + ROLES.index(actor_role)] = 1.0
    for rank, count in rank_counts.items():
        values[23 + rank - 1] = count / (2 if rank == 13 else rank)
    for relative_slot in range(player_count):
        absolute_seat = (actor_seat + relative_slot) % player_count
        offset = 49 + relative_slot * 12
        values[offset] = 1.0
        values[offset + 1] = (
            sum(rank_counts.values()) / 20 if relative_slot == 0 else 1 / 20
        )
        values[offset + 4] = 1.0 if relative_slot == 0 else 0.0
        role = role_for_index(absolute_seat, player_count)
        values[offset + 7 + ROLES.index(role)] = 1.0
    values[169] = 1.0
    return values


class V3PpoPipelineTests(unittest.TestCase):
    def runner_arguments(self) -> argparse.Namespace:
        return argparse.Namespace(
            data=["data/*.ndjson"],
            behavior_model="behavior-model.json",
            output="models/v3-safe-run-001",
            results_dir="returned/v3-safe-run-001",
            epochs=12,
            batch_size=4096,
            learning_rate=1.0e-4,
            weight_decay=1.0e-5,
            gamma=1.0,
            gae_lambda=1.0,
            skip_forced_policy_time=True,
            terminal_rank_auxiliary_coefficient=0.0,
            rollout_temperature=2.5,
            clip_coefficient=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            max_gradient_norm=0.5,
            target_kl=0.015,
            binding_tolerance=2.0e-5,
            behavior_binding_batch_size=8192,
            loader_workers=7,
            seed=202608061,
            device="cuda",
        )

    def runner_config(self, args: argparse.Namespace) -> dict:
        return {
            "format": "dalmuti-v3-ppo-gpu-run-config",
            "version": 2,
            "algorithm": expected_algorithm(args),
            "allowedTerminalRankAuxiliaryCoefficients": [0, 0.05],
            "determinism": dict(EXPECTED_DETERMINISM),
            "pathPolicy": dict(EXPECTED_PATH_POLICY),
        }

    @staticmethod
    def write_records(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def create_fixture(self, root: Path) -> tuple[Path, Path]:
        model_path = root / "behavior-v3.json"
        model = V3ActionConditionedActorCriticNetwork(
            observation_features=172,
            observation_schema_version=2,
            actor_observation_hidden_sizes=(2,),
            actor_action_hidden_sizes=(2,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(2,),
        )
        for parameter in model.parameters():
            parameter.data.zero_()
        export_v3_action_conditioned_json(model, model_path)
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        rollout_path = root / "rollout.ndjson"
        manifest = {
            "type": "manifest",
            "format": "dalmuti-v3-ppo-ndjson",
            "formatVersion": 1,
            "createdAt": "2026-08-01T00:00:00.000Z",
            "environment": {
                "game": "DALMUTI",
                "rules": "project-house-rules-v1",
                "playerCount": 4,
                "actsPerEpisode": 1,
                "episodes": 1,
                "initialSeed": 1,
                "rolloutMode": "league",
                "opponentPolicies": ["normal"],
                "nonCardDecisions": "normal bot policy",
                "reward": "actorTerminal ? (roundChipAward - 2) / 2 : 0",
                "learnerSeats": (
                    "approximately half; only behavior-model decisions are samples"
                ),
                "opponentMix": {
                    "normalFraction": 1,
                    "trainedModelFraction": 0,
                    "trainedModelSelection": "uniform",
                    "trainedModels": [],
                },
                "collection": {
                    "mode": "fixed-episodes",
                    "requestedEpisodes": 1,
                },
            },
            "behaviorModel": {
                "sha256": model_sha256,
                "format": "dalmuti-action-conditioned-actor-critic",
                "observationSchemaVersion": 2,
                "observationFeatures": 172,
                "actionCatalogueVersion": 1,
            },
            "behaviorPolicy": {
                "sampling": "softmax",
                "temperature": 1.25,
                "logProbabilityBinding": (
                    "recomputed from behavior model over exactly "
                    "legalMaskHex at this temperature"
                ),
            },
            "observation": {
                "version": 2,
                "featureCount": 172,
                "privacy": (
                    "own private hand plus public state only; "
                    "opponent hands excluded"
                ),
            },
            "actionSpace": {
                "catalogueVersion": 1,
                "size": V3_ACTION_COUNT,
                "catalogue": [dict(action) for action in V3_ACTION_CATALOGUE],
                "actionFeatures": V3_ACTION_FEATURE_COUNT,
                "actionFeatureLayout": list(V3_ACTION_FEATURE_LAYOUT),
                "encodedActionFeatures": [
                    list(features) for features in V3_ACTION_FEATURES
                ],
                "legalMaskEncoding": {
                    "field": "legalMaskHex",
                    "lowercaseHexDigits": 59,
                    "bitOrder": (
                        "action index i = bit (i % 4) of hex digit "
                        "floor(i / 4)"
                    ),
                },
            },
            "sampleBindings": {
                "observationSchemaVersion": 2,
                "actionCatalogueVersion": 1,
                "policyVersion": f"sha256:{model_sha256}",
                "legalActionIndices": (
                    "unique ascending indices exactly equal to legalMaskHex"
                ),
                "forced": (
                    "true exactly when legalActionIndices has length one"
                ),
            },
        }
        def common(actor: int, finish_place: int) -> dict:
            episode_id = "v3-league-p4-episode-1"
            actor_id = f"player-{actor}"
            return {
                "type": "sample",
                "trajectoryId": f"{episode_id}:round-1:{actor_id}",
                "episodeId": episode_id,
                "round": 1,
                "actorId": actor_id,
                "actorSeat": actor - 1,
                "actorRole": role_for_index(actor - 1, 4),
                "observationSchemaVersion": 2,
                "actionCatalogueVersion": 1,
                "oldValue": 0.0,
                "finishPlace": finish_place,
                "policyVersion": f"sha256:{model_sha256}",
            }

        samples = [
            {
                **common(1, 1),
                "step": 0,
                "observation": observation({1: 1, 2: 1}, actor_seat=0),
                "legalActionIndices": [2, 5],
                "legalMaskHex": mask_hex([2, 5]),
                "actionIndex": 2,
                "oldLogProbability": -math.log(2),
                "reward": 0.0,
                "terminal": False,
                "forced": False,
            },
            {
                **common(2, 2),
                "step": 1,
                "observation": observation({1: 1, 2: 1}, actor_seat=1),
                "legalActionIndices": [2, 5],
                "legalMaskHex": mask_hex([2, 5]),
                "actionIndex": 2,
                "oldLogProbability": -math.log(2),
                "reward": 0.0,
                "terminal": False,
                "forced": False,
            },
            {
                **common(1, 1),
                "step": 2,
                "observation": observation({1: 1}, actor_seat=0),
                "legalActionIndices": [2],
                "legalMaskHex": mask_hex([2]),
                "actionIndex": 2,
                "oldLogProbability": 0.0,
                "reward": 1.0,
                "terminal": True,
                "forced": True,
            },
            {
                **common(2, 2),
                "step": 3,
                "observation": observation({1: 1}, actor_seat=1),
                "legalActionIndices": [2],
                "legalMaskHex": mask_hex([2]),
                "actionIndex": 2,
                "oldLogProbability": 0.0,
                "reward": 0.5,
                "terminal": True,
                "forced": True,
            },
        ]
        summary = {
            "type": "summary",
            "episodes": 1,
            "learnerSamples": 4,
            "environmentDecisions": 4,
            "forcedSamples": 2,
            "nonForcedSamples": 2,
            "behaviorModelSha256": model_sha256,
            "samplingTemperature": 1.25,
            "targetNonForcedDecisions": None,
            "opponentModelSha256": [],
            "opponentSeatAssignments": {
                "normal": 2,
                "byModelSha256": {},
            },
        }
        rollout_path.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n"
                for record in (manifest, *samples, summary)
            ),
            encoding="utf-8",
        )
        return model_path, rollout_path

    def create_parallel_fixture(
        self, root: Path
    ) -> tuple[Path, tuple[Path, Path]]:
        model, rollout_p4 = self.create_fixture(root)
        records = [
            json.loads(line)
            for line in rollout_p4.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["environment"]["playerCount"] = 5
        records[0]["environment"]["initialSeed"] = 2
        for index, record in enumerate(records[1:-1]):
            actor = int(record["actorId"].removeprefix("player-"))
            actor_seat = actor - 1
            episode_id = "v3-league-p5-episode-1"
            record["episodeId"] = episode_id
            record["trajectoryId"] = (
                f"{episode_id}:round-1:{record['actorId']}"
            )
            record["actorRole"] = role_for_index(actor_seat, 5)
            record["observation"] = observation(
                {1: 1, 2: 1} if index < 2 else {1: 1},
                player_count=5,
                actor_seat=actor_seat,
            )
        records[-1]["opponentSeatAssignments"]["normal"] = 3
        rollout_p5 = root / "rollout-p5.ndjson"
        self.write_records(rollout_p5, records)
        return model, (rollout_p4, rollout_p5)

    def test_strict_loader_recomputes_behavior_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            source_files: list[dict[str, object]] = []
            loaded = load_v3_ppo_rollouts(
                [str(rollout)],
                behavior_model_path=model,
                source_files_out=source_files,
            )
            self.assertEqual(loaded.legal_masks.shape, (4, 236))
            self.assertEqual(loaded.behavior_temperature, 1.25)
            np.testing.assert_allclose(loaded.returns, [1.0, 0.5, 1.0, 0.5])
            self.assertEqual(
                source_files,
                [
                    {
                        "path": str(rollout.resolve()),
                        "bytes": rollout.stat().st_size,
                        "sha256": hashlib.sha256(rollout.read_bytes()).hexdigest(),
                    }
                ],
            )

            records = [
                json.loads(line)
                for line in rollout.read_text(encoding="utf-8").splitlines()
            ]
            records[1]["oldLogProbability"] += 0.1
            bad = root / "bad.ndjson"
            bad.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "log-probability binding"):
                load_v3_ppo_rollouts([str(bad)], behavior_model_path=model)

            records[1]["oldLogProbability"] -= 0.1
            records[1]["legalActionIndices"] = [2]
            records[1]["legalMaskHex"] = mask_hex([2])
            records[1]["forced"] = True
            bad.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "encoded observation"):
                load_v3_ppo_rollouts([str(bad)], behavior_model_path=model)

    def test_parallel_loader_is_array_and_order_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollouts = self.create_parallel_fixture(root)
            baseline_sources: list[dict[str, object]] = []
            parallel_sources: list[dict[str, object]] = []
            baseline = load_v3_ppo_rollouts(
                [str(path) for path in rollouts],
                behavior_model_path=model,
                terminal_rank_auxiliary_coefficient=0.05,
                loader_workers=1,
                source_files_out=baseline_sources,
            )
            parallel = load_v3_ppo_rollouts(
                [str(path) for path in reversed(rollouts)],
                behavior_model_path=model,
                terminal_rank_auxiliary_coefficient=0.05,
                loader_workers=2,
                source_files_out=parallel_sources,
            )
            for field in (
                "observations",
                "legal_masks",
                "actions",
                "old_log_probabilities",
                "old_values",
                "rewards",
                "rank_auxiliary_rewards",
                "effective_rewards",
                "terminals",
                "forced",
                "advantages",
                "returns",
                "trajectory_ids",
            ):
                with self.subTest(field=field):
                    np.testing.assert_array_equal(
                        getattr(parallel, field), getattr(baseline, field)
                    )
            self.assertEqual(parallel.files, baseline.files)
            self.assertEqual(
                parallel.behavior_model_sha256,
                baseline.behavior_model_sha256,
            )
            self.assertEqual(
                parallel.behavior_temperature, baseline.behavior_temperature
            )
            self.assertEqual(
                parallel.trajectory_count, baseline.trajectory_count
            )
            self.assertEqual(parallel_sources, baseline_sources)
            baseline_report = build_v3_ppo_data_verification(
                baseline,
                source_files=baseline_sources,
                gamma=1.0,
                gae_lambda=1.0,
                rollout_temperature=1.25,
                binding_tolerance=2.0e-5,
            )
            parallel_report = build_v3_ppo_data_verification(
                parallel,
                source_files=parallel_sources,
                gamma=1.0,
                gae_lambda=1.0,
                rollout_temperature=1.25,
                binding_tolerance=2.0e-5,
            )
            self.assertEqual(parallel_report, baseline_report)
            self.assertEqual(parallel_report["version"], 2)

            duplicate = root / "duplicate.ndjson"
            duplicate.write_bytes(rollouts[0].read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicated across"):
                load_v3_ppo_rollouts(
                    [str(rollouts[0]), str(duplicate)],
                    loader_workers=2,
                )

            for invalid in (0, -1, 1.5, True):
                with self.subTest(invalid_loader_workers=invalid):
                    with self.assertRaisesRegex(ValueError, "loader workers"):
                        load_v3_ppo_rollouts(
                            [str(rollouts[0])], loader_workers=invalid
                        )

    def test_behavior_binding_runtime_options_preserve_cpu_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            baseline = load_v3_ppo_rollouts(
                [str(rollout)],
                behavior_model_path=model,
            )

            for batch_size in (1, 3, 64):
                loaded = load_v3_ppo_rollouts(
                    [str(rollout)],
                    behavior_model_path=model,
                    behavior_binding_device="cpu",
                    behavior_binding_batch_size=batch_size,
                )
                np.testing.assert_array_equal(
                    loaded.old_log_probabilities,
                    baseline.old_log_probabilities,
                )
                np.testing.assert_array_equal(
                    loaded.old_values, baseline.old_values
                )
                np.testing.assert_array_equal(
                    loaded.advantages, baseline.advantages
                )
                np.testing.assert_array_equal(loaded.returns, baseline.returns)

            for invalid in (0, -1, 1.5, True):
                with self.subTest(invalid_batch_size=invalid):
                    with self.assertRaisesRegex(ValueError, "batch size"):
                        load_v3_ppo_rollouts(
                            [str(rollout)],
                            behavior_model_path=model,
                            behavior_binding_batch_size=invalid,
                        )
            for invalid in ("meta", "not-a-device"):
                with self.subTest(invalid_device=invalid):
                    with self.assertRaisesRegex(ValueError, "device"):
                        load_v3_ppo_rollouts(
                            [str(rollout)],
                            behavior_model_path=model,
                            behavior_binding_device=invalid,
                        )

    def test_behavior_binding_tolerance_is_stable_across_batch_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            records = [
                json.loads(line)
                for line in rollout.read_text(encoding="utf-8").splitlines()
            ]
            records[1]["oldLogProbability"] += 5.0e-6
            perturbed = root / "perturbed.ndjson"
            self.write_records(perturbed, records)

            for batch_size in (1, 64):
                with self.subTest(batch_size=batch_size):
                    for _ in range(2):
                        load_v3_ppo_rollouts(
                            [str(perturbed)],
                            behavior_model_path=model,
                            binding_tolerance=1.0e-5,
                            behavior_binding_device="cpu",
                            behavior_binding_batch_size=batch_size,
                        )
                    with self.assertRaisesRegex(
                        ValueError, "log-probability binding"
                    ):
                        load_v3_ppo_rollouts(
                            [str(perturbed)],
                            behavior_model_path=model,
                            binding_tolerance=1.0e-6,
                            behavior_binding_device="cpu",
                            behavior_binding_batch_size=batch_size,
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_behavior_binding_matches_cpu_acceptance_deterministically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            records = [
                json.loads(line)
                for line in rollout.read_text(encoding="utf-8").splitlines()
            ]
            records[1]["oldLogProbability"] += 5.0e-6
            perturbed = root / "perturbed.ndjson"
            self.write_records(perturbed, records)
            cpu = load_v3_ppo_rollouts(
                [str(perturbed)],
                behavior_model_path=model,
                binding_tolerance=1.0e-5,
                behavior_binding_device="cpu",
                behavior_binding_batch_size=2,
            )
            for _ in range(2):
                cuda = load_v3_ppo_rollouts(
                    [str(perturbed)],
                    behavior_model_path=model,
                    binding_tolerance=1.0e-5,
                    behavior_binding_device="cuda",
                    behavior_binding_batch_size=64,
                )
                np.testing.assert_array_equal(cuda.advantages, cpu.advantages)
                np.testing.assert_array_equal(cuda.returns, cpu.returns)
            with self.assertRaisesRegex(ValueError, "log-probability binding"):
                load_v3_ppo_rollouts(
                    [str(perturbed)],
                    behavior_model_path=model,
                    binding_tolerance=1.0e-6,
                    behavior_binding_device="cuda",
                    behavior_binding_batch_size=64,
                )

    @unittest.skipIf(torch.cuda.is_available(), "CUDA is available")
    def test_explicit_cuda_binding_fails_closed_without_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model, rollout = self.create_fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                load_v3_ppo_rollouts(
                    [str(rollout)],
                    behavior_model_path=model,
                    behavior_binding_device="cuda",
                    behavior_binding_batch_size=64,
                )

    def test_semantics_contract_matches_schema(self) -> None:
        schema = json.loads((ROOT / "v3-ppo-schema.json").read_text(encoding="utf-8"))
        self.assertEqual(
            schema["rolloutSemanticsContract"], V3_PPO_SEMANTICS_CONTRACT
        )
        self.assertEqual(
            schema["rolloutSemanticsContractSha256"],
            V3_PPO_SEMANTICS_CONTRACT_SHA256,
        )
        encoded = json.dumps(
            schema["rolloutSemanticsContract"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            V3_PPO_SEMANTICS_CONTRACT_SHA256,
        )

    def test_loader_rejects_semantic_and_count_corruption(self) -> None:
        mutations = [
            (
                "environment",
                lambda records: records[0]["environment"].__setitem__(
                    "game", "FORGED"
                ),
                "environment semantics",
            ),
            (
                "sample-binding",
                lambda records: records[0]["sampleBindings"].__setitem__(
                    "forced", "untrusted"
                ),
                "sample binding",
            ),
            (
                "trajectory",
                lambda records: records[1].__setitem__(
                    "trajectoryId", "forged-trajectory"
                ),
                "trajectoryId provenance",
            ),
            (
                "reward",
                lambda records: records[3].__setitem__("reward", 0.0),
                "reward does not match",
            ),
            (
                "observation",
                lambda records: records[1]["observation"].__setitem__(0, 1.0),
                "observation player count",
            ),
            (
                "summary-environment-decisions",
                lambda records: records[-1].__setitem__(
                    "environmentDecisions", 3
                ),
                "rollout summary",
            ),
            (
                "summary-forced",
                lambda records: records[-1].__setitem__("forcedSamples", True),
                "summary.forcedSamples",
            ),
            (
                "opponent-assignments",
                lambda records: records[-1]["opponentSeatAssignments"].__setitem__(
                    "normal", 1
                ),
                "rollout summary",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            original = [
                json.loads(line)
                for line in rollout.read_text(encoding="utf-8").splitlines()
            ]
            for name, mutate, error in mutations:
                with self.subTest(name=name):
                    records = json.loads(json.dumps(original))
                    mutate(records)
                    bad = root / f"bad-{name}.ndjson"
                    self.write_records(bad, records)
                    with self.assertRaisesRegex(ValueError, error):
                        load_v3_ppo_rollouts(
                            [str(bad)], behavior_model_path=model
                        )

    def test_loader_and_cli_reject_non_finite_or_corrupt_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            original = [
                json.loads(line)
                for line in rollout.read_text(encoding="utf-8").splitlines()
            ]

            valid_verification = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_ppo_data.py"),
                    "--data",
                    str(rollout),
                    "--behavior-model",
                    str(model),
                    "--rollout-temperature",
                    "1.25",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                valid_verification.returncode, 0, valid_verification.stderr
            )
            verification_payload = json.loads(valid_verification.stdout)
            self.assertEqual(verification_payload["version"], 2)
            self.assertEqual(
                verification_payload["rolloutSemanticsContract"]["sha256"],
                V3_PPO_SEMANTICS_CONTRACT_SHA256,
            )
            self.assertEqual(
                verification_payload["sourceFiles"],
                [
                    {
                        "path": str(rollout.resolve()),
                        "bytes": rollout.stat().st_size,
                        "sha256": hashlib.sha256(rollout.read_bytes()).hexdigest(),
                    }
                ],
            )

            huge = json.loads(json.dumps(original))
            huge[1]["oldValue"] = 1.0e100
            huge_path = root / "huge.ndjson"
            self.write_records(huge_path, huge)
            with self.assertRaisesRegex(ValueError, "float32 storage"):
                load_v3_ppo_rollouts([str(huge_path)])

            corrupt = json.loads(json.dumps(original))
            corrupt[3]["reward"] = -1.0
            corrupt_path = root / "corrupt.ndjson"
            self.write_records(corrupt_path, corrupt)
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_ppo_data.py"),
                    "--data",
                    str(corrupt_path),
                    "--behavior-model",
                    str(model),
                    "--rollout-temperature",
                    "1.25",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("reward does not match", verified.stderr)

    def test_gpu_runner_binds_algorithm_and_fresh_disjoint_paths(self) -> None:
        args = self.runner_arguments()
        config = self.runner_config(args)
        validate_run_config(config, args)
        self.assertTrue(sys.dont_write_bytecode)
        environment = strict_python_environment(
            args.seed,
            {
                "PYTHONDONTWRITEBYTECODE": "0",
                "PYTHONHASHSEED": "wrong",
                "CUBLAS_WORKSPACE_CONFIG": "wrong",
            },
        )
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["PYTHONHASHSEED"], str(args.seed))
        self.assertEqual(
            environment["CUBLAS_WORKSPACE_CONFIG"],
            ":4096:8",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_id, output, results = resolve_fresh_run_paths(
                root,
                args.output,
                args.results_dir,
                config["pathPolicy"],
            )
            self.assertEqual(run_id, "v3-safe-run-001")
            self.assertEqual(output.parent, root / "models")
            self.assertEqual(results.parent, root / "returned")
            with self.assertRaisesRegex(ValueError, "same run ID"):
                resolve_fresh_run_paths(
                    root,
                    args.output,
                    "returned/different-run",
                    config["pathPolicy"],
                )
            with self.assertRaisesRegex(ValueError, "relative path"):
                resolve_fresh_run_paths(
                    root,
                    "../escaped",
                    args.results_dir,
                    config["pathPolicy"],
                )
            output.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "must not already exist"):
                resolve_fresh_run_paths(
                    root,
                    args.output,
                    args.results_dir,
                    config["pathPolicy"],
                )

        changed = argparse.Namespace(**vars(args))
        changed.epochs = 11
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            validate_run_config(config, changed)
        changed = argparse.Namespace(**vars(args))
        changed.terminal_rank_auxiliary_coefficient = 0.1
        with self.assertRaisesRegex(ValueError, "not an approved"):
            validate_run_config(config, changed)

    def test_gpu_runner_protects_exact_manifest_bound_inputs(self) -> None:
        args = self.runner_arguments()
        config = self.runner_config(args)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "data").mkdir()
            behavior = root / "behavior-model.json"
            rollout = root / "data" / "p4.ndjson"
            behavior.write_bytes(b"model\n")
            rollout.write_bytes(b"rollout\n")

            def entry(path: Path) -> dict:
                return {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            manifest = {
                "rollouts": [{"filename": rollout.name}],
                "files": [entry(behavior), entry(rollout)],
            }
            resolved_behavior, resolved_data = resolve_protected_inputs(
                root, args, config, manifest
            )
            self.assertEqual(resolved_behavior, behavior)
            self.assertEqual(resolved_data, [rollout])
            snapshot = protected_input_snapshot([behavior, rollout])
            assert_protected_inputs_unchanged(snapshot)
            rollout.write_bytes(b"mutated\n")
            with self.assertRaisesRegex(RuntimeError, "changed during run"):
                assert_protected_inputs_unchanged(snapshot)

            args.data = ["data/missing.ndjson"]
            with self.assertRaisesRegex(ValueError, "every and only"):
                resolve_protected_inputs(root, args, config, manifest)

    def test_preflight_records_the_deterministic_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "hardware-report.json"
            environment = {
                **os.environ,
                "PYTHONHASHSEED": "202608061",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "preflight.py"),
                    "--device",
                    "cpu",
                    "--deterministic",
                    "--seed",
                    "202608061",
                    "--output",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            runtime = report["deterministicRuntime"]
            self.assertTrue(runtime["algorithmsEnabled"])
            self.assertFalse(runtime["warnOnly"])
            self.assertEqual(runtime["seed"], 202608061)
            self.assertEqual(runtime["pythonHashSeed"], "202608061")
            self.assertFalse(runtime["cudnnBenchmark"])
            self.assertFalse(runtime["cudaMatmulAllowTf32"])

            repeated = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "preflight.py"),
                    "--device",
                    "cpu",
                    "--deterministic",
                    "--seed",
                    "202608061",
                    "--output",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("FileExistsError", repeated.stderr)

    def test_packaging_runs_after_training_log_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "training.log"
            log_path.write_text("training complete\n", encoding="utf-8")
            environment = {
                **os.environ,
                "DALMUTI_RUNNER_TEST_OUTPUT": "package-output",
            }
            run_after_sealing_log(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['DALMUTI_RUNNER_TEST_OUTPUT'])",
                ],
                log_path,
                environment=environment,
            )
            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("python", contents.lower())
            self.assertNotIn("package-output", contents)

    def test_fused_training_fails_before_checkpoint_on_binding_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            original = [
                json.loads(line)
                for line in rollout.read_text(encoding="utf-8").splitlines()
            ]

            def train(data: Path, output: Path) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "train_v3_ppo.py"),
                        "--data",
                        str(data),
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
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

            for label, field in (
                ("log-probability", "oldLogProbability"),
                ("value", "oldValue"),
            ):
                with self.subTest(label=label):
                    records = json.loads(json.dumps(original))
                    records[1][field] += 0.1
                    tampered = root / f"tampered-{label}.ndjson"
                    self.write_records(tampered, records)
                    output = root / f"failed-{label}"
                    trained = train(tampered, output)
                    self.assertNotEqual(trained.returncode, 0)
                    self.assertIn("binding mismatch", trained.stderr)
                    self.assertFalse((output / "checkpoint.pt").exists())
                    self.assertFalse((output / "checkpoints").exists())

            stale_output = root / "stale-verification"
            stale_output.mkdir()
            (stale_output / "data-verification.json").write_text(
                "{}\n", encoding="utf-8"
            )
            stale = train(rollout, stale_output)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("must not already exist", stale.stderr)
            self.assertFalse((stale_output / "checkpoint.pt").exists())

    def test_one_epoch_cpu_train_verify_and_package_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, rollout = self.create_fixture(root)
            output = root / "fresh-v3-run"
            standalone_report = root / "standalone-data-verification.json"
            verify = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_ppo_data.py"),
                    "--data",
                    str(rollout),
                    "--behavior-model",
                    str(model),
                    "--rollout-temperature",
                    "1.25",
                    "--output",
                    str(standalone_report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)
            training = subprocess.run(
                [
                    sys.executable,
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
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(training.returncode, 0, training.stderr)
            self.assertEqual(
                standalone_report.read_bytes(),
                (output / "data-verification.json").read_bytes(),
            )
            metadata = json.loads(
                (output / "v3-ppo-metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                metadata["deterministicRuntime"]["algorithmsEnabled"]
            )
            self.assertFalse(metadata["deterministicRuntime"]["warnOnly"])
            self.assertIsNone(metadata["sourceProvenance"])
            self.assertNotIn("data_verification_output", metadata["arguments"])
            self.assertNotIn(
                "behavior_binding_batch_size", metadata["arguments"]
            )
            self.assertEqual(metadata["sourceData"][0]["path"], "data/rollout.ndjson")
            self.assertEqual(
                metadata["sourceData"][0]["sha256"],
                hashlib.sha256(rollout.read_bytes()).hexdigest(),
            )

            deterministic_repeat = root / "fresh-v3-run-deterministic-repeat"
            repeated_training = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train_v3_ppo.py"),
                    "--data",
                    str(rollout),
                    "--behavior-model",
                    str(model),
                    "--output",
                    str(deterministic_repeat),
                    "--data-verification-output",
                    str(deterministic_repeat / "data-verification.json"),
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
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                repeated_training.returncode, 0, repeated_training.stderr
            )
            self.assertEqual(
                (output / "v3-actor-critic-weights.json").read_bytes(),
                (
                    deterministic_repeat / "v3-actor-critic-weights.json"
                ).read_bytes(),
            )
            (output / "hardware-report.json").write_text("{}\n", encoding="utf-8")
            (output / "training.log").write_text(
                training.stdout, encoding="utf-8"
            )
            results = root / "returned"
            packaged = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "package_v3_ppo_results.py"),
                    "--model-dir",
                    str(output),
                    "--results-dir",
                    str(results),
                    "--allow-legacy-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            archive = results / "fresh-v3-run-result.zip"
            checksum_path = results / "fresh-v3-run-result.zip.sha256"
            expected = checksum_path.read_text(encoding="utf-8").split()[0]
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)
            with zipfile.ZipFile(archive) as packaged_zip:
                names = set(packaged_zip.namelist())
            self.assertIn("v3-actor-critic-weights.json", names)
            self.assertIn(
                "checkpoints/epoch-01/v3-actor-critic-weights.json", names
            )
            extracted = root / "verified-result"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_ppo_results.py"),
                    "--archive",
                    str(archive),
                    "--checksum",
                    str(checksum_path),
                    "--extract-dir",
                    str(extracted),
                    "--allow-legacy-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertTrue(
                (extracted / "v3-actor-critic-weights.json").is_file()
            )

            repeated = subprocess.run(
                [
                    sys.executable,
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
                    "--rollout-temperature",
                    "1.25",
                    "--device",
                    "cpu",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("must not already exist", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
