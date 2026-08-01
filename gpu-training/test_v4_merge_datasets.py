from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_COUNT,
)
from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
    load_v4_dataset_npz,
)
from v4_merge_datasets import (
    DAGGER_PREPARATION_FORMAT,
    MERGED_PREPARATION_FORMAT,
    NORMAL_PREPARATION_FORMAT,
    PPO_PREPARATION_FORMAT,
    merge_v4_datasets,
)
from v4_env import (
    PRIVILEGED_STATE_LAYOUT,
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
)
from v4_model import V4ActorConfig, V4CriticConfig
from v4_ppo_advantages import (
    MERGED_BASELINE_MIN_REFERENCES,
    MERGED_GLOBAL_SCALE_FLOOR,
    MERGED_PPO_ADVANTAGE_CONTRACT,
    merged_ppo_advantage_array_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalogue_sha256() -> str:
    return hashlib.sha256(json.dumps(
        {
            "version": V3_ACTION_CATALOGUE_VERSION,
            "catalogue": [dict(item) for item in V3_ACTION_CATALOGUE],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _write_checksum(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(f"{_sha256(path)}  {path.name}\n", encoding="ascii")
    return sidecar


def _actor(max_players: int, max_history: int, *, d_model: int = 24) -> V4ActorConfig:
    return V4ActorConfig(
        max_players=max_players,
        max_history=max_history,
        d_model=d_model,
        layers=1,
        heads=4,
        feedforward=48,
        action_hidden=16,
    )


def _critic() -> V4CriticConfig:
    return V4CriticConfig(
        privileged_features=512,
        d_model=32,
        hidden_layers=1,
        action_hidden=16,
    )


def _standard_arrays(
    actor: V4ActorConfig,
    critic: V4CriticConfig,
    lengths: list[int],
    *,
    player_count: int,
    role: int,
    act: int,
    marker: float,
) -> dict[str, np.ndarray]:
    count = len(lengths)
    time_steps = max(lengths)
    prefix = (count, time_steps)
    arrays: dict[str, np.ndarray] = {
        "global_features": np.zeros((*prefix, actor.global_features), np.float32),
        "rank_features": np.zeros((*prefix, actor.rank_tokens, actor.rank_features), np.float32),
        "player_features": np.zeros((*prefix, actor.max_players, actor.player_features), np.float32),
        "player_mask": np.zeros((*prefix, actor.max_players), np.bool_),
        "memory_trace_features": np.zeros((*prefix, actor.memory_tokens, actor.memory_features), np.float32),
        "history_features": np.zeros((*prefix, actor.max_history, actor.history_features), np.float32),
        "history_mask": np.zeros((*prefix, actor.max_history), np.bool_),
        "legal_masks": np.zeros((*prefix, V3_ACTION_COUNT), np.bool_),
        "actions": np.zeros(prefix, np.int64),
        "expert_actions": np.zeros(prefix, np.int64),
        "old_action_log_probs": np.zeros(prefix, np.float32),
        "advantages": np.zeros(prefix, np.float32),
        "rewards": np.zeros(prefix, np.float32),
        "dones": np.zeros(prefix, np.bool_),
        "valid_masks": np.zeros(prefix, np.bool_),
        "privileged_states": np.zeros((*prefix, critic.privileged_features), np.float32),
    }
    for trajectory, length in enumerate(lengths):
        valid = slice(0, length)
        arrays["valid_masks"][trajectory, valid] = True
        arrays["global_features"][trajectory, valid, 0] = (player_count - 4) / 6.0
        arrays["global_features"][trajectory, valid, 1] = math.tanh((act - 1) / 10.0)
        arrays["global_features"][trajectory, valid, 2 + role] = 1.0
        arrays["rank_features"][trajectory, valid, 0, 0] = marker + trajectory
        arrays["player_mask"][trajectory, valid, :player_count] = True
        arrays["player_features"][trajectory, valid, :player_count, 0] = marker
        history_length = min(actor.max_history, trajectory + 1)
        arrays["history_mask"][trajectory, valid, :history_length] = True
        arrays["history_features"][trajectory, valid, :history_length, 0] = marker
        arrays["legal_masks"][trajectory, valid, 0] = True
        arrays["legal_masks"][trajectory, valid, 1] = True
        arrays["actions"][trajectory, valid] = 0
        arrays["expert_actions"][trajectory, valid] = 1
        arrays["old_action_log_probs"][trajectory, valid] = -0.5
        arrays["advantages"][trajectory, valid] = marker / 10.0
        arrays["rewards"][trajectory, length - 1] = marker
        arrays["dones"][trajectory, length - 1] = True
        arrays["privileged_states"][trajectory, valid, 0] = marker + 100.0
    return arrays


def _dataset(
    arrays: dict[str, np.ndarray], actor: V4ActorConfig, critic: V4CriticConfig
) -> V4TrajectoryDataset:
    tensors: dict[str, torch.Tensor] = {}
    boolean = {"player_mask", "history_mask", "legal_masks", "dones", "valid_masks"}
    integer = {"actions", "expert_actions"}
    for field in fields(V4TrajectoryTensors):
        tensor = torch.from_numpy(arrays[field.name])
        tensors[field.name] = (
            tensor.bool() if field.name in boolean
            else tensor.long() if field.name in integer
            else tensor.float()
        )
    return V4TrajectoryDataset(V4TrajectoryTensors(**tensors), actor, critic)


def _write_prepared(
    path: Path,
    preparation: str,
    *,
    ids: list[str],
    lengths: list[int],
    max_players: int,
    max_history: int,
    player_count: int,
    role: int,
    act: int,
    marker: float,
    d_model: int = 24,
    terminal_markers: list[float] | None = None,
    match_clusters: list[str] | None = None,
) -> Path:
    actor = _actor(max_players, max_history, d_model=d_model)
    critic = _critic()
    arrays = _standard_arrays(
        actor, critic, lengths,
        player_count=player_count, role=role, act=act, marker=marker,
    )
    count, time_steps = arrays["actions"].shape
    valid = arrays["valid_masks"]
    arrays["trajectory_ids"] = np.asarray(ids, dtype=np.str_)
    if preparation in {NORMAL_PREPARATION_FORMAT, DAGGER_PREPARATION_FORMAT}:
        arrays["finish_places"] = np.zeros((count, time_steps), np.int16)
        arrays["environment_terminals"] = np.zeros((count, time_steps), np.bool_)
        for index, length in enumerate(lengths):
            arrays["finish_places"][index, length - 1] = index + 1
    if preparation == NORMAL_PREPARATION_FORMAT:
        raw_sha = hashlib.sha256(f"normal-{marker}".encode()).hexdigest()
        arrays["source_steps"] = np.full((count, time_steps), -1, np.int64)
        for index, length in enumerate(lengths):
            arrays["source_steps"][index, :length] = np.arange(length)
        arrays["trajectory_input_sha256s"] = np.asarray([raw_sha] * count, dtype=np.str_)
        source_hashes = {
            "actionCatalogue": _catalogue_sha256(),
            "actorObservationContract": hashlib.sha256(b"actor").hexdigest(),
            "privilegedCriticContract": hashlib.sha256(b"ts-critic").hexdigest(),
        }
        metadata: dict[str, object] = {
            "format": V4_DATASET_FORMAT,
            "version": V4_DATASET_VERSION,
            "preparationFormat": preparation,
            "preparationVersion": 1,
            "actorConfig": actor.to_dict(),
            "criticConfig": critic.to_dict(),
            "inputs": [{
                "sha256": raw_sha,
                "format": "dalmuti-v4-normal-warmstart-ndjson",
                "formatVersion": 1,
                "playerCount": player_count,
                "actsPerEpisode": 1,
                "sourceHashes": source_hashes,
            }],
            "privilegedCriticExportAllowed": False,
            "trajectoryIds": ids,
            "auxiliaryArrays": ["finish_places", "environment_terminals", "source_steps"],
        }
    elif preparation == DAGGER_PREPARATION_FORMAT:
        env_sha = hashlib.sha256(b"env").hexdigest()
        arrays["candidate_actions"] = np.full((count, time_steps), -1, np.int64)
        arrays["behavior_sources"] = np.full((count, time_steps), -1, np.int8)
        arrays["forced_masks"] = np.zeros((count, time_steps), np.bool_)
        arrays["source_decision_indices"] = np.full((count, time_steps), -1, np.int64)
        for index, length in enumerate(lengths):
            arrays["candidate_actions"][index, :length] = 1
            arrays["behavior_sources"][index, :length] = index % 2
            arrays["source_decision_indices"][index, :length] = np.arange(length)
        arrays["trajectory_player_counts"] = np.asarray([player_count] * count, np.int16)
        arrays["trajectory_roles"] = np.asarray([role] * count, np.int8)
        arrays["trajectory_acts"] = np.asarray([act] * count, np.int16)
        arrays["trajectory_actor_ids"] = np.arange(count, dtype=np.int16)
        arrays["trajectory_match_indices"] = np.arange(count, dtype=np.int32)
        arrays["trajectory_match_seeds"] = np.arange(100, 100 + count, dtype=np.uint32)
        metadata = {
            "format": V4_DATASET_FORMAT,
            "version": V4_DATASET_VERSION,
            "preparationFormat": preparation,
            "preparationVersion": 1,
            "actorConfig": actor.to_dict(),
            "criticConfig": critic.to_dict(),
            "collection": {
                "algorithm": "DAgger",
                "expert": "exact-v4-env-Normal",
                "expertLabelForEveryDecision": True,
            },
            "privacy": {
                "actorPublicOnly": True,
                "opponentPhysicalHandsExcluded": True,
                "taxCardIdentitiesExcluded": True,
                "privilegedCriticStateSeparate": True,
                "privilegedCriticExportAllowed": False,
            },
            "environmentBinding": {
                "normalExpertCallback": "DalmutiScalarEnv.normal_action",
                "v4EnvSha256": env_sha,
            },
            "privilegedCriticLayout": {
                "id": PRIVILEGED_STATE_LAYOUT_ID,
                "sha256": PRIVILEGED_STATE_LAYOUT_SHA256,
                "layout": PRIVILEGED_STATE_LAYOUT,
                "featureCount": 512,
                "matchesTypescriptNormalContract": True,
            },
            "modelBinding": {
                "criticExcluded": True,
                "bundleManifestSha256": hashlib.sha256(b"manifest").hexdigest(),
                "actorCheckpointSha256": hashlib.sha256(b"actor.pt").hexdigest(),
            },
            "sourceHashes": {"gpu-training/v4_env.py": env_sha},
            "auxiliaryArrays": [
                "candidate_actions", "behavior_sources", "forced_masks",
                "finish_places", "environment_terminals", "source_decision_indices",
                "trajectory_ids", "trajectory_player_counts", "trajectory_roles",
                "trajectory_acts", "trajectory_actor_ids", "trajectory_match_indices",
                "trajectory_match_seeds",
            ],
        }
    elif preparation == PPO_PREPARATION_FORMAT:
        if terminal_markers is None:
            terminal_markers = [marker] * count
        if len(terminal_markers) != count:
            raise ValueError("terminal_markers must match ids")
        if match_clusters is None:
            match_clusters = [f"ppo-cluster-{identifier}" for identifier in ids]
        if len(match_clusters) != count:
            raise ValueError("match_clusters must match ids")
        env_sha = hashlib.sha256(b"ppo-env").hexdigest()
        shape = (count, time_steps)
        arrays.update({
            "raw_returns": np.zeros(shape, np.float32),
            "baseline_values": np.zeros(shape, np.float32),
            "raw_advantages": np.zeros(shape, np.float32),
            "advantage_scales": np.ones(shape, np.float32),
            "baseline_tiers": np.full(shape, -1, np.int8),
            "baseline_reference_counts": np.zeros(shape, np.int32),
            "selected_action_probabilities": np.zeros(shape, np.float64),
            "policy_entropies": np.zeros(shape, np.float32),
            "terminal_chip_awards": np.zeros(shape, np.int8),
            "forced_masks": np.zeros(shape, np.bool_),
            "source_decision_indices": np.full(shape, -1, np.int64),
            "trajectory_player_counts": np.asarray([player_count] * count, np.int16),
            "trajectory_roles": np.asarray([role] * count, np.int8),
            "trajectory_acts": np.asarray([act] * count, np.int16),
            "trajectory_actor_ids": np.asarray(
                [index % player_count for index in range(count)], np.int16
            ),
            "trajectory_match_indices": np.arange(count, dtype=np.int32),
            "trajectory_match_seeds": np.arange(200, 200 + count, dtype=np.uint32),
            "trajectory_match_clusters": np.asarray(match_clusters, dtype=np.str_),
            "trajectory_finish_places": np.asarray(
                [index % player_count + 1 for index in range(count)], np.int16
            ),
        })
        probability = math.exp(-0.5)
        for index, length in enumerate(lengths):
            terminal_marker = float(terminal_markers[index])
            chip_award = int(round(terminal_marker * 2.0 + 2.0))
            if chip_award not in range(5) or not math.isclose(
                terminal_marker, (chip_award - 2.0) / 2.0
            ):
                raise ValueError("terminal markers must be exact normalized chip rewards")
            valid_slice = (index, slice(0, length))
            arrays["rewards"][index, :] = 0.0
            arrays["rewards"][index, length - 1] = terminal_marker
            arrays["raw_returns"][valid_slice] = terminal_marker
            arrays["raw_advantages"][valid_slice] = terminal_marker
            arrays["advantages"][valid_slice] = terminal_marker
            arrays["baseline_tiers"][valid_slice] = 0
            arrays["baseline_reference_counts"][valid_slice] = 2
            arrays["selected_action_probabilities"][valid_slice] = probability
            arrays["policy_entropies"][valid_slice] = 0.25
            arrays["source_decision_indices"][valid_slice] = np.arange(length)
            arrays["terminal_chip_awards"][index, length - 1] = chip_award
        metadata = {
            "format": V4_DATASET_FORMAT,
            "version": V4_DATASET_VERSION,
            "preparationFormat": preparation,
            "preparationVersion": 1,
            "actorConfig": actor.to_dict(),
            "criticConfig": critic.to_dict(),
            "collection": {
                "algorithm": "on-policy PPO league rollout",
                "exactOldLogProbabilityForEveryLearnerDecision": True,
                "exactNormalExpertLabelForEveryLearnerDecision": True,
            },
            "returnsAndAdvantages": {
                "monteCarloGamma": 1.0,
                "standardized": True,
            },
            "privacy": {
                "actorPublicOnly": True,
                "opponentPhysicalHandsExcluded": True,
                "taxCardIdentitiesExcluded": True,
                "privilegedCriticStateSeparate": True,
                "privilegedCriticExportAllowed": False,
            },
            "environmentBinding": {
                "normalExpertCallback": "DalmutiScalarEnv.normal_action",
                "v4EnvSha256": env_sha,
            },
            "privilegedCriticBinding": {
                "layoutId": PRIVILEGED_STATE_LAYOUT_ID,
                "layoutSha256": PRIVILEGED_STATE_LAYOUT_SHA256,
                "layout": PRIVILEGED_STATE_LAYOUT,
                "featureCount": 512,
                "environmentSourceSha256": env_sha,
                "actorExportAllowed": False,
            },
            "modelBinding": {
                "criticExcluded": True,
                "bundleManifestSha256": hashlib.sha256(b"ppo-manifest").hexdigest(),
                "actorCheckpointSha256": hashlib.sha256(b"ppo-actor.pt").hexdigest(),
            },
            "sourceHashes": {"gpu-training/v4_env.py": env_sha},
            "auxiliaryArrays": sorted(set(arrays) - {field.name for field in fields(V4TrajectoryTensors)}),
        }
    else:
        raise AssertionError("test helper supports Normal, DAgger, or PPO")
    dataset = _dataset(arrays, actor, critic)
    metadata.update({
        "fingerprint": dataset.fingerprint,
        "trajectoryCount": count,
        "sampleCount": int(valid.sum()),
        "maxTimeSteps": time_steps,
    })
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(path, **arrays)
    return _write_checksum(path)


def _rewrite(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    _write_checksum(path)


class V4MergeDatasetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _inputs(self) -> tuple[Path, Path]:
        normal = self.root / "normal.npz"
        dagger = self.root / "dagger.npz"
        _write_prepared(
            normal, NORMAL_PREPARATION_FORMAT,
            ids=["normal-a", "normal-b"], lengths=[3, 2],
            max_players=6, max_history=3, player_count=5, role=2, act=1,
            marker=1.25,
        )
        _write_prepared(
            dagger, DAGGER_PREPARATION_FORMAT,
            ids=["dagger-a", "dagger-b"], lengths=[2, 4],
            max_players=8, max_history=5, player_count=7, role=4, act=2,
            marker=2.5,
        )
        return normal, dagger

    def _ppo_shard(
        self,
        name: str,
        indices: list[int],
        rewards: list[float],
        clusters: list[str],
    ) -> Path:
        path = self.root / f"{name}.npz"
        _write_prepared(
            path,
            PPO_PREPARATION_FORMAT,
            ids=[f"ppo-global-{index:03d}" for index in indices],
            lengths=[2 + index % 2 for index in indices],
            max_players=8,
            max_history=5,
            player_count=7,
            role=1,
            act=2,
            marker=0.0,
            terminal_markers=[rewards[index] for index in indices],
            match_clusters=[clusters[index] for index in indices],
        )
        return path

    def test_normal_and_dagger_merge_is_deterministic_and_loader_compatible(self) -> None:
        normal, dagger = self._inputs()
        first = self.root / "first.npz"
        second = self.root / "second.npz"
        result = merge_v4_datasets([normal, dagger], first)
        reversed_result = merge_v4_datasets([dagger, normal], second)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(
            Path(f"{first}.metadata.json").read_bytes(),
            Path(f"{second}.metadata.json").read_bytes(),
        )
        self.assertEqual(result.npz_sha256, reversed_result.npz_sha256)
        self.assertEqual(result.trajectories, 4)
        self.assertEqual(result.samples, 11)
        loaded = load_v4_dataset_npz(first)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded.actor_config.max_players, 8)
        self.assertEqual(loaded.actor_config.max_history, 5)
        self.assertEqual(loaded.tensors.actions.shape, (4, 4))
        self.assertEqual(loaded.fingerprint, result.fingerprint)

        recursive = self.root / "recursive.npz"
        recursive_result = merge_v4_datasets(first, recursive)
        self.assertEqual(recursive_result.fingerprint, result.fingerprint)
        self.assertEqual(load_v4_dataset_npz(recursive).fingerprint, result.fingerprint)

        with np.load(first, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            source_order = [item["sha256"] for item in metadata["inputs"]]
            self.assertEqual(source_order, sorted([_sha256(normal), _sha256(dagger)]))
            self.assertEqual(metadata["preparationFormat"], MERGED_PREPARATION_FORMAT)
            self.assertEqual(
                metadata["lossEligibility"]["eligibleSampleCounts"],
                {"behaviorCloning": 11, "ppo": 0, "critic": 0},
            )
            self.assertEqual(metadata["lossEligibility"]["ppoBehaviorActorSha256s"], [])
            ids = archive["trajectory_ids"].tolist()
            expected = (
                ["normal-a", "normal-b"]
                if _sha256(normal) < _sha256(dagger)
                else ["dagger-a", "dagger-b"]
            )
            self.assertEqual(ids[:2], expected)
            for index, identifier in enumerate(ids):
                valid = archive["valid_masks"][index]
                terminal = archive["dones"][index]
                self.assertEqual(int((terminal & valid).sum()), 1)
                self.assertTrue(terminal[np.flatnonzero(valid)[-1]])
                np.testing.assert_array_equal(
                    archive["bc_eligible_masks"][index], valid
                )
                self.assertFalse(archive["ppo_eligible_masks"][index].any())
                self.assertFalse(archive["critic_eligible_masks"][index].any())
                self.assertFalse(archive["player_mask"][index, valid, 8:].any())
                self.assertFalse(archive["history_mask"][index, valid, 5:].any())
                if identifier.startswith("normal"):
                    self.assertTrue(np.all(archive["candidate_actions"][index, valid] == -1))
                    self.assertTrue(np.all(archive["behavior_sources"][index, valid] == 0))
                    self.assertTrue(np.all(archive["source_decision_indices"][index, valid] == -1))
                else:
                    self.assertTrue(np.all(archive["source_steps"][index, valid] == -1))
            self.assertEqual(metadata["balance"]["byPlayerCount"]["5"]["trajectories"], 2)
            self.assertEqual(metadata["balance"]["byPlayerCount"]["7"]["trajectories"], 2)

        self.assertEqual(
            Path(f"{first}.sha256").read_text(encoding="ascii").split()[0],
            _sha256(first),
        )
        metadata_path = Path(f"{first}.metadata.json")
        self.assertEqual(
            Path(f"{metadata_path}.sha256").read_text(encoding="ascii").split()[0],
            _sha256(metadata_path),
        )

    def test_padding_preserves_every_valid_source_value(self) -> None:
        normal, dagger = self._inputs()
        output = self.root / "merged.npz"
        merge_v4_datasets([normal, dagger], output)
        source_by_sha = {_sha256(normal): normal, _sha256(dagger): dagger}
        with np.load(output, allow_pickle=False) as merged:
            ids = merged["trajectory_ids"].tolist()
            source_shas = merged["trajectory_source_npz_sha256s"].tolist()
            for index, (identifier, source_sha) in enumerate(zip(ids, source_shas, strict=True)):
                source_path = source_by_sha[source_sha]
                with np.load(source_path, allow_pickle=False) as source:
                    source_index = source["trajectory_ids"].tolist().index(identifier)
                    length = int(source["valid_masks"][source_index].sum())
                    for name in (
                        "global_features", "rank_features", "memory_trace_features",
                        "legal_masks", "actions", "expert_actions",
                        "old_action_log_probs", "advantages", "rewards", "dones",
                        "valid_masks", "privileged_states",
                    ):
                        np.testing.assert_array_equal(
                            merged[name][index, :length], source[name][source_index, :length]
                        )
                    source_players = source["player_features"].shape[2]
                    source_history = source["history_features"].shape[2]
                    np.testing.assert_array_equal(
                        merged["player_features"][index, :length, :source_players],
                        source["player_features"][source_index, :length],
                    )
                    np.testing.assert_array_equal(
                        merged["history_features"][index, :length, :source_history],
                        source["history_features"][source_index, :length],
                    )

    def test_ppo_auxiliaries_and_canonical_layout_merge(self) -> None:
        normal, _ = self._inputs()
        ppo = self.root / "ppo.npz"
        _write_prepared(
            ppo, PPO_PREPARATION_FORMAT,
            ids=["ppo-a", "ppo-b"], lengths=[2, 3],
            max_players=8, max_history=5, player_count=7, role=1, act=2,
            marker=0.5,
        )
        output = self.root / "normal-ppo.npz"
        result = merge_v4_datasets([ppo, normal], output)
        loaded = load_v4_dataset_npz(output)
        self.assertEqual(loaded.fingerprint, result.fingerprint)
        with np.load(output, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            self.assertEqual(
                metadata["privilegedCriticLayout"]["sha256"],
                PRIVILEGED_STATE_LAYOUT_SHA256,
            )
            self.assertEqual(
                metadata["privilegedCriticLayout"]["layout"],
                PRIVILEGED_STATE_LAYOUT,
            )
            ids = archive["trajectory_ids"].tolist()
            for index, identifier in enumerate(ids):
                valid = archive["valid_masks"][index]
                if identifier.startswith("ppo"):
                    np.testing.assert_array_equal(
                        archive["ppo_eligible_masks"][index], valid
                    )
                    np.testing.assert_array_equal(
                        archive["critic_eligible_masks"][index], valid
                    )
                    np.testing.assert_allclose(
                        archive["old_action_log_probs"][index, valid],
                        np.log(archive["selected_action_probabilities"][index, valid]),
                        atol=2.0e-6,
                    )
                    self.assertTrue(
                        np.all(archive["trajectory_match_clusters"][index] != "")
                    )
                else:
                    self.assertFalse(archive["ppo_eligible_masks"][index].any())
                    self.assertFalse(archive["critic_eligible_masks"][index].any())
                    self.assertTrue(np.all(
                        archive["selected_action_probabilities"][index, valid] == 0.0
                    ))
                    self.assertEqual(archive["trajectory_match_clusters"][index], "")
            self.assertEqual(
                metadata["lossEligibility"]["eligibleSampleCounts"],
                {"behaviorCloning": 10, "ppo": 5, "critic": 5},
            )
            expected_actor_sha = hashlib.sha256(b"ppo-actor.pt").hexdigest()
            self.assertEqual(
                metadata["lossEligibility"]["ppoBehaviorActorSha256s"],
                [expected_actor_sha],
            )
        recursive = self.root / "normal-ppo-recursive.npz"
        merge_v4_datasets(output, recursive)
        with np.load(recursive, allow_pickle=False) as archive:
            recursive_metadata = json.loads(str(archive["metadata_json"].item()))
        self.assertEqual(
            recursive_metadata["lossEligibility"]["ppoBehaviorActorSha256s"],
            [hashlib.sha256(b"ppo-actor.pt").hexdigest()],
        )

    def test_global_ppo_recompute_is_shard_partition_invariant(self) -> None:
        rewards = [-1.0, -0.5, 0.0, 0.5, 1.0] * 4
        clusters = [f"match-{index:03d}" for index in range(len(rewards))]
        contiguous = [
            self._ppo_shard("contiguous-a", list(range(0, 9)), rewards, clusters),
            self._ppo_shard("contiguous-b", list(range(9, 20)), rewards, clusters),
        ]
        interleaved = [
            self._ppo_shard("interleaved-a", list(range(0, 20, 2)), rewards, clusters),
            self._ppo_shard("interleaved-b", list(range(1, 20, 2)), rewards, clusters),
        ]
        first = self.root / "partition-contiguous.npz"
        reversed_output = self.root / "partition-contiguous-reversed.npz"
        second = self.root / "partition-interleaved.npz"
        first_result = merge_v4_datasets(contiguous, first)
        reversed_result = merge_v4_datasets(
            list(reversed(contiguous)), reversed_output
        )
        merge_v4_datasets(interleaved, second)

        self.assertEqual(first.read_bytes(), reversed_output.read_bytes())
        self.assertEqual(first_result.npz_sha256, reversed_result.npz_sha256)

        def by_id(path: Path) -> tuple[dict[str, tuple[float, float, float, int]], dict[str, object]]:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                output: dict[str, tuple[float, float, float, int]] = {}
                for index, identifier in enumerate(archive["trajectory_ids"].tolist()):
                    if not identifier.startswith("ppo-global-"):
                        continue
                    output[identifier] = (
                        float(archive["baseline_values"][index, 0]),
                        float(archive["raw_advantages"][index, 0]),
                        float(archive["advantage_scales"][index, 0]),
                        int(archive["baseline_reference_counts"][index, 0]),
                    )
                return output, metadata

        contiguous_values, contiguous_metadata = by_id(first)
        interleaved_values, interleaved_metadata = by_id(second)
        self.assertEqual(contiguous_values, interleaved_values)
        contract = contiguous_metadata["returnsAndAdvantages"]
        self.assertEqual(contract["format"], MERGED_PPO_ADVANTAGE_CONTRACT)
        self.assertEqual(contract["version"], 2)
        self.assertEqual(
            contract["minimumReferenceCount"], MERGED_BASELINE_MIN_REFERENCES
        )
        self.assertGreaterEqual(
            contract["globalPopulationScale"], MERGED_GLOBAL_SCALE_FLOOR
        )
        terminal_raw_advantages = []
        for index, reward in enumerate(rewards):
            expected_baseline = float(np.mean(rewards[:index] + rewards[index + 1:]))
            identifier = f"ppo-global-{index:03d}"
            self.assertAlmostEqual(
                contiguous_values[identifier][0], expected_baseline, places=6
            )
            terminal_raw_advantages.append(reward - expected_baseline)
        expected_scale = max(
            float(np.std(np.asarray(terminal_raw_advantages), ddof=0)),
            MERGED_GLOBAL_SCALE_FLOOR,
        )
        self.assertAlmostEqual(
            contract["globalPopulationScale"], expected_scale, places=12
        )
        self.assertEqual(contract["fallbackCounts"]["same-player-count-role-act"], 20)
        self.assertTrue(all(value[3] == 19 for value in contiguous_values.values()))
        self.assertEqual(
            contract["globalPopulationScale"],
            interleaved_metadata["returnsAndAdvantages"]["globalPopulationScale"],
        )

    def test_global_ppo_baseline_excludes_the_whole_own_match_cluster(self) -> None:
        external = [-1.0, -0.5, 0.0, 0.5, 1.0] * 4
        rewards_a = [1.0, -1.0, *external]
        rewards_b = [-0.5, 0.5, *external]
        clusters = ["target-match", "target-match"] + [
            f"external-{index:03d}" for index in range(len(external))
        ]
        input_a = self._ppo_shard(
            "leakage-a", list(range(len(rewards_a))), rewards_a, clusters
        )
        output_a = self.root / "leakage-out-a.npz"
        merge_v4_datasets(input_a, output_a)
        input_b = self._ppo_shard(
            "leakage-b", list(range(len(rewards_b))), rewards_b, clusters
        )
        output_b = self.root / "leakage-out-b.npz"
        merge_v4_datasets(input_b, output_b)

        def target_baselines(path: Path) -> tuple[float, float]:
            with np.load(path, allow_pickle=False) as archive:
                ids = archive["trajectory_ids"].tolist()
                values = []
                for identifier in ("ppo-global-000", "ppo-global-001"):
                    index = ids.index(identifier)
                    values.append(float(archive["baseline_values"][index, 0]))
                    self.assertEqual(
                        int(archive["baseline_reference_counts"][index, 0]), 20
                    )
                return values[0], values[1]

        self.assertEqual(target_baselines(output_a), target_baselines(output_b))
        self.assertEqual(target_baselines(output_a), (0.0, 0.0))

    def test_merged_ppo_recomputed_arrays_load_and_coherent_tampering_is_rejected(self) -> None:
        rewards = [-1.0, -0.5, 0.0, 0.5, 1.0] * 4
        clusters = [f"tamper-match-{index:03d}" for index in range(20)]
        source = self._ppo_shard(
            "tamper-source", list(range(20)), rewards, clusters
        )
        output = self.root / "tamper-merged.npz"
        merge_v4_datasets(source, output)
        loaded = load_v4_dataset_npz(output)
        self.assertEqual(int(loaded.loss_eligibility.ppo.sum()), int(sum(2 + i % 2 for i in range(20))))
        with np.load(source, allow_pickle=False) as direct_archive:
            direct_metadata = json.loads(str(direct_archive["metadata_json"].item()))
        self.assertEqual(direct_metadata["preparationVersion"], 1)
        with np.load(output, allow_pickle=False) as merged_archive:
            merged_metadata = json.loads(str(merged_archive["metadata_json"].item()))
        self.assertEqual(
            merged_metadata["returnsAndAdvantages"]["format"],
            MERGED_PPO_ADVANTAGE_CONTRACT,
        )

        old_v1 = self.root / "old-merged-v1.npz"
        merge_v4_datasets(source, old_v1)

        def downgrade_contract(arrays: dict[str, np.ndarray]) -> None:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["returnsAndAdvantages"]["format"] = (
                "dalmuti-v4-global-merged-lomo-advantages-v1"
            )
            metadata["returnsAndAdvantages"]["version"] = 1
            arrays["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            )

        _rewrite(old_v1, downgrade_contract)
        with self.assertRaisesRegex(ValueError, "contract is missing or incompatible"):
            load_v4_dataset_npz(old_v1)

        def corrupt_coherently(arrays: dict[str, np.ndarray]) -> None:
            ppo = arrays["ppo_eligible_masks"]
            row = int(np.flatnonzero(ppo.any(axis=1))[0])
            mask = ppo[row]
            arrays["baseline_values"][row, mask] += np.float32(0.25)
            arrays["raw_advantages"][row, mask] = (
                arrays["raw_returns"][row, mask]
                - arrays["baseline_values"][row, mask]
            )
            arrays["advantages"][row, mask] = (
                arrays["raw_advantages"][row, mask]
                / arrays["advantage_scales"][row, mask]
            )
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["returnsAndAdvantages"]["arrayBindingSha256"] = (
                merged_ppo_advantage_array_sha256(arrays)
            )
            arrays["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            )

        _rewrite(output, corrupt_coherently)
        with self.assertRaisesRegex(ValueError, "stale or has been tampered"):
            load_v4_dataset_npz(output)

    def test_collision_config_checksum_fingerprint_and_semantics_rejection(self) -> None:
        normal, dagger = self._inputs()

        legacy = self.root / "legacy-dagger.npz"
        _write_prepared(
            legacy, DAGGER_PREPARATION_FORMAT,
            ids=["legacy-a"], lengths=[1], max_players=8, max_history=5,
            player_count=7, role=4, act=2, marker=2.25,
        )

        def remove_layout(arrays: dict[str, np.ndarray]) -> None:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            del metadata["privilegedCriticLayout"]
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))

        _rewrite(legacy, remove_layout)
        with self.assertRaisesRegex(ValueError, "privileged critic layout"):
            merge_v4_datasets([normal, legacy], self.root / "legacy-out.npz")

        wrong_layout = self.root / "wrong-layout-dagger.npz"
        _write_prepared(
            wrong_layout, DAGGER_PREPARATION_FORMAT,
            ids=["wrong-layout-a"], lengths=[1], max_players=8, max_history=5,
            player_count=7, role=4, act=2, marker=2.375,
        )

        def alter_layout(arrays: dict[str, np.ndarray]) -> None:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["privilegedCriticLayout"]["layout"]["players"]["offset"] = 30
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))

        _rewrite(wrong_layout, alter_layout)
        with self.assertRaisesRegex(ValueError, "privileged critic layout"):
            merge_v4_datasets([normal, wrong_layout], self.root / "wrong-layout-out.npz")

        collision = self.root / "collision.npz"
        _write_prepared(
            collision, DAGGER_PREPARATION_FORMAT,
            ids=["normal-a"], lengths=[1], max_players=8, max_history=5,
            player_count=7, role=4, act=2, marker=3.0,
        )
        with self.assertRaisesRegex(ValueError, "duplicate trajectory ID"):
            merge_v4_datasets([normal, collision], self.root / "collision-out.npz")

        drift = self.root / "drift.npz"
        _write_prepared(
            drift, DAGGER_PREPARATION_FORMAT,
            ids=["drift-a"], lengths=[1], max_players=8, max_history=5,
            player_count=7, role=4, act=2, marker=4.0, d_model=32,
        )
        with self.assertRaisesRegex(ValueError, "actor configuration drift"):
            merge_v4_datasets([normal, drift], self.root / "drift-out.npz")

        checksum = Path(f"{dagger}.sha256")
        checksum.write_text("0" * 64 + f"  {dagger.name}\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "does not match"):
            merge_v4_datasets([normal, dagger], self.root / "checksum-out.npz")
        _write_checksum(dagger)

        def corrupt_tensor(arrays: dict[str, np.ndarray]) -> None:
            arrays["rewards"][0, 0] += 0.125

        _rewrite(dagger, corrupt_tensor)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            merge_v4_datasets([normal, dagger], self.root / "fingerprint-out.npz")

        _write_prepared(
            dagger, DAGGER_PREPARATION_FORMAT,
            ids=["dagger-c"], lengths=[2], max_players=8, max_history=5,
            player_count=7, role=4, act=2, marker=2.75,
        )

        def corrupt_semantics(arrays: dict[str, np.ndarray]) -> None:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["collection"]["expert"] = "approximate-policy"
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))

        _rewrite(dagger, corrupt_semantics)
        with self.assertRaisesRegex(ValueError, "semantics"):
            merge_v4_datasets([normal, dagger], self.root / "semantics-out.npz")

    def test_output_is_immutable(self) -> None:
        normal, dagger = self._inputs()
        output = self.root / "immutable.npz"
        merge_v4_datasets([normal, dagger], output)
        snapshots = {
            path: path.read_bytes()
            for path in (
                output,
                Path(f"{output}.metadata.json"),
                Path(f"{output}.sha256"),
                Path(f"{output}.metadata.json.sha256"),
            )
        }
        with self.assertRaises(FileExistsError):
            merge_v4_datasets([normal, dagger], output)
        for path, payload in snapshots.items():
            self.assertEqual(path.read_bytes(), payload)

    def test_training_smoke_compatibility(self) -> None:
        try:
            from v4_train import V4TrainingConfig, train_v4
        except ImportError:
            self.skipTest("V4 torch training stack is unavailable")
        normal, dagger = self._inputs()
        output = self.root / "trainable.npz"
        merge_v4_datasets([normal, dagger], output)
        dataset = load_v4_dataset_npz(output)
        result = train_v4(
            dataset,
            self.root / "training",
            V4TrainingConfig(
                epochs=1,
                batch_size=2,
                gradient_accumulation=1,
                bc_weight=1.0,
                ppo_weight=0.0,
                critic_weight=0.0,
                q_boost_coefficient=0.0,
                entropy_coefficient=0.0,
                amp=False,
                checkpoint_every=1,
            ),
            device="cpu",
        )
        self.assertEqual(result["datasetFingerprint"], dataset.fingerprint)
        self.assertTrue((self.root / "training" / "candidate" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
