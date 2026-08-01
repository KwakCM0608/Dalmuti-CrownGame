import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from non_card_action_conditioned import (
    REVOLUTION_ACTION_COUNT,
    REVOLUTION_ACTION_FEATURE_COUNT,
    REVOLUTION_OBSERVATION_FEATURE_COUNT,
    TAX_RETURN_ACTION_COUNT,
    TAX_RETURN_ACTION_FEATURE_COUNT,
    TAX_RETURN_ACTION_FEATURES,
    TAX_RETURN_OBSERVATION_FEATURE_COUNT,
    RevolutionActionConditionedActorCriticNetwork,
    TaxReturnActionConditionedActorCriticNetwork,
)
from non_card_counterfactual_dataset import (
    canonical_world_key,
    deterministic_validation_membership,
    load_non_card_counterfactuals,
)
from package_non_card_results import (
    package_result_directory,
    verify_result_archive,
)
from train_non_card_counterfactual import (
    TrainingOptions,
    apply_policy_temperature,
    apply_utility_target,
    supervised_loss,
    train_non_card_models,
    validate_options,
)
from verify_non_card_results import verify_result_directory


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _soft_targets(utilities: list[float], temperature: float = 1.0) -> list[float]:
    maximum = max(utilities)
    exponentials = [math.exp((value - maximum) / temperature) for value in utilities]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _decision_act_utility(place: int, player_count: int = 4) -> float:
    if place == 1:
        award = 4
    elif place == 2:
        award = 3
    elif place == player_count - 1:
        award = 1
    elif place == player_count:
        award = 0
    else:
        award = 2
    return (award - 2) / 2


def _fill_public_slots(
    observation: list[float], actor_seat: int, actor_hand_count: int
) -> None:
    roles = ["great-dalmuti", "lesser-dalmuti", "lesser-peon", "great-peon"]
    role_names = [
        "great-dalmuti",
        "lesser-dalmuti",
        "merchant",
        "lesser-peon",
        "great-peon",
    ]
    for slot in range(4):
        offset = 21 + slot * 8
        role = roles[(actor_seat + slot) % 4]
        observation[offset] = 1.0
        observation[offset + 1] = (
            actor_hand_count / 20 if slot == 0 else 5.0 / 20.0
        )
        observation[offset + 2] = 0.0
        observation[offset + 3 + role_names.index(role)] = 1.0


def _tax_observation() -> list[float]:
    observation = [0.0] * TAX_RETURN_OBSERVATION_FEATURE_COUNT
    observation[2] = 3.0 / 20.0
    observation[4] = 1.0  # lesser Dalmuti returns one card
    observation[8] = 1.0  # rank 1 x1
    observation[9] = 0.5  # rank 2 x1 of two deck copies
    observation[20] = 0.5  # joker x1 of two deck copies
    observation[101] = 1.0
    _fill_public_slots(observation, actor_seat=1, actor_hand_count=3)
    return observation


def _revolution_observation(great_peon: bool) -> list[float]:
    observation = [0.0] * REVOLUTION_OBSERVATION_FEATURE_COUNT
    observation[2] = 2.0 / 20.0
    observation[7 if great_peon else 3] = 1.0
    observation[20] = 1.0
    _fill_public_slots(
        observation,
        actor_seat=3 if great_peon else 0,
        actor_hand_count=2,
    )
    return observation


def _record(
    decision: str,
    episode: int,
    *,
    match_seed_base: int,
    run_label: str,
    acts: int,
    temperature: float,
) -> dict:
    episode_id = f"{run_label}-episode-{episode:03d}"
    pre_hash = _sha(f"{run_label}-{decision}-pre-{episode}")
    paired_world = f"sha256:{_sha(f'{run_label}-world-{episode}') }"
    if decision == "tax-return":
        observation = _tax_observation()
        legal_indices = [0, 1, 12]
        legal_mask = [index in legal_indices for index in range(TAX_RETURN_ACTION_COUNT)]
        actor_role = "lesser-dalmuti"
        actor_seat = 1
        utilities = [float((episode + position) % 4) for position in range(3)]
        features = [list(TAX_RETURN_ACTION_FEATURES[index]) for index in legal_indices]
        metadata = {"playerCount": 4, "actorHandCount": 3, "returnCount": 1}
        namespace = "taxReturn"
    else:
        great_peon = episode % 2 == 0
        observation = _revolution_observation(great_peon)
        legal_indices = [0, 1]
        legal_mask = [True, True]
        actor_role = "great-peon" if great_peon else "great-dalmuti"
        actor_seat = 3 if great_peon else 0
        utilities = [float(episode % 3), float((episode + 1) % 3)]
        features = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0] if great_peon else [0.0, 1.0, 0.0],
        ]
        metadata = {"playerCount": 4, "actorHandCount": 2}
        namespace = "revolution"
    center = sum(utilities) / len(utilities)
    probabilities = _soft_targets(utilities, temperature)
    best_position = max(range(len(utilities)), key=lambda index: utilities[index])
    actions = []
    for position, action_index in enumerate(legal_indices):
        actions.append(
            {
                "actionIndex": action_index,
                "actionFeatures": features[position],
                "pairedWorldId": paired_world,
                "terminalActorUtility": utilities[position],
                "decisionActUtility": _decision_act_utility(1 + position),
                "terminalFinishPlaceInDecisionAct": 1 + position,
                "meanUtility": utilities[position],
                "centeredUtility": utilities[position] - center,
                "uncertainty": {
                    "sampleStandardDeviation": 0.0,
                    "standardError": 0.0,
                },
                "softTargetProbability": probabilities[position],
            }
        )
    return {
        "type": "counterfactual-decision",
        "sampleId": f"{decision}:{pre_hash}",
        "decision": decision,
        "episodeId": episode_id,
        "matchSeed": match_seed_base + episode,
        "playerCount": 4,
        "acts": acts,
        "round": 1,
        "actorId": f"p{actor_seat}",
        "actorSeat": actor_seat,
        "actorRole": actor_role,
        "decisionKey": f"{episode_id}:round-1:actor-p{actor_seat}",
        "observationSchemaVersion": 1,
        "actionCatalogueVersion": 1,
        "observation": observation,
        "legalMask": legal_mask,
        "legalActionIndices": legal_indices,
        "baselineActionIndex": legal_indices[0],
        "metadata": metadata,
        "pairing": {
            "pairedWorldId": paired_world,
            "preDecisionSha256": pre_hash,
            "continuationPolicy": "normal-deterministic",
            "forcedOverrideNamespace": namespace,
            "rootActionCoverage": "all-legal-actions-exactly-once",
        },
        "utility": {
            "definition": "terminal-cumulative-chip-score",
            "centeredAcrossLegalActions": True,
        },
        "targetBuilder": "training/non-card-search-targets.ts#buildPairedCounterfactualTargets",
        "targetSampleCount": 1,
        "bestActionIndex": legal_indices[best_position],
        "actions": actions,
    }


def _write_fixture(
    path: Path,
    episodes: int = 24,
    *,
    match_seed_base: int = 800000,
    run_label: str = "fixture",
    acts: int = 3,
    temperature: float = 1.0,
) -> None:
    manifest = {
        "type": "manifest",
        "format": "dalmuti-non-card-counterfactual-ndjson",
        "version": 1,
        "createdAt": "2026-08-01T00:00:00.000Z",
        "observationSchemaVersion": 1,
        "actionCatalogueVersions": {"taxReturn": 1, "revolution": 1},
        "featureDimensions": {
            "taxReturn": {
                "observation": TAX_RETURN_OBSERVATION_FEATURE_COUNT,
                "action": TAX_RETURN_ACTION_FEATURE_COUNT,
                "catalogue": TAX_RETURN_ACTION_COUNT,
            },
            "revolution": {
                "observation": REVOLUTION_OBSERVATION_FEATURE_COUNT,
                "action": REVOLUTION_ACTION_FEATURE_COUNT,
                "catalogue": REVOLUTION_ACTION_COUNT,
            },
        },
        "collection": {
            "playerCounts": [4],
            "episodesPerPlayerCount": episodes,
            "acts": acts,
            "initialSeed": match_seed_base,
            "matchSeedDerivation": "initialSeed + zero-based index over ascending playerCount then episode",
            "decisionKinds": ["tax-return", "revolution"],
            "policyTemperature": temperature,
            "maxDecisions": None,
            "baselineNonCardHooks": {},
            "continuationPolicy": "normal-deterministic",
            "resumeAllowed": False,
        },
        "privacy": {
            "observation": "encoded-actor-hand-and-public-state-only",
            "opponentCardIdentitiesIncluded": False,
            "physicalCardIdsIncluded": False,
        },
    }
    records = [
        _record(
            decision,
            episode,
            match_seed_base=match_seed_base,
            run_label=run_label,
            acts=acts,
            temperature=temperature,
        )
        for episode in range(episodes)
        for decision in ("tax-return", "revolution")
    ]
    content = b""
    for value in [manifest, *records]:
        content += (
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    action_evaluations = episodes * 5
    summary = {
        "type": "summary",
        "baselineMatches": episodes,
        "decisionsDiscovered": episodes * 2,
        "decisionsWritten": episodes * 2,
        "actionEvaluations": action_evaluations,
        "stoppedAtMaxDecisions": False,
        "counts": {
            "byDecision": {
                "tax-return": {
                    "discovered": episodes,
                    "written": episodes,
                    "actionEvaluations": episodes * 3,
                },
                "revolution": {
                    "discovered": episodes,
                    "written": episodes,
                    "actionEvaluations": episodes * 2,
                },
            },
            "byPlayerCount": {
                "4": {
                    "baselineMatches": episodes,
                    "decisionsWritten": episodes * 2,
                    "actionEvaluations": action_evaluations,
                }
            },
        },
        "hashes": {
            "algorithm": "sha256",
            "contentBeforeSummary": hashlib.sha256(content).hexdigest(),
            "contentBeforeSummaryBytes": len(content),
            "scope": "UTF-8 NDJSON bytes for manifest and decision records, including newlines",
        },
    }
    content += (
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(content)


class NonCardCounterfactualPipelineTests(unittest.TestCase):
    def test_streaming_loader_validates_and_groups_entire_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.ndjson"
            _write_fixture(path)
            first = load_non_card_counterfactuals(
                [str(path)], validation_fraction=0.3, split_seed=77
            )
            second = load_non_card_counterfactuals(
                [str(path)], validation_fraction=0.3, split_seed=77
            )
            self.assertEqual(first.group_split_key, "canonicalWorldKey")
            self.assertIsNotNone(first.tax_return)
            self.assertIsNotNone(first.revolution)
            self.assertEqual(
                first.tax_return.train.sample_ids,
                second.tax_return.train.sample_ids,
            )
            self.assertTrue(
                np.all(
                    first.tax_return.train.legal_masks[
                        np.arange(len(first.tax_return.train)),
                        first.tax_return.train.baseline_actions,
                    ]
                )
            )
            train_worlds = set(first.tax_return.train.world_keys) | set(
                first.revolution.train.world_keys
            )
            validation_worlds = set(first.tax_return.validation.world_keys) | set(
                first.revolution.validation.world_keys
            )
            self.assertFalse(train_worlds & validation_worlds)
            for episode in range(24):
                world_key = canonical_world_key(
                    player_count=4,
                    acts=3,
                    match_seed=800000 + episode,
                    continuation_policy="normal-deterministic",
                )
                expected_validation = deterministic_validation_membership(
                    world_key, split_seed=77, validation_fraction=0.3
                )
                self.assertEqual(world_key in validation_worlds, expected_validation)

    def test_loader_rejects_content_tampering_and_cross_file_world_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.ndjson"
            _write_fixture(original, episodes=8)
            tampered = root / "tampered.ndjson"
            lines = original.read_text(encoding="utf-8").splitlines()
            summary = json.loads(lines[-1])
            summary["hashes"]["contentBeforeSummary"] = "0" * 64
            lines[-1] = json.dumps(summary, separators=(",", ":"))
            data = ("\n".join(lines) + "\n").encode("utf-8")
            tampered.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "content SHA-256 mismatch"):
                load_non_card_counterfactuals([str(tampered)])

            duplicate = root / "duplicate.ndjson"
            shutil.copyfile(original, duplicate)
            with self.assertRaisesRegex(ValueError, "canonical hidden world overlaps"):
                load_non_card_counterfactuals([str(original), str(duplicate)])

    def test_multifile_acts_and_temperature_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "temperature-1.ndjson"
            second = root / "temperature-2.ndjson"
            different_acts = root / "acts-4.ndjson"
            _write_fixture(
                first,
                episodes=12,
                match_seed_base=810000,
                run_label="temperature-one",
                temperature=1.0,
            )
            _write_fixture(
                second,
                episodes=12,
                match_seed_base=820000,
                run_label="temperature-two",
                temperature=2.0,
            )
            _write_fixture(
                different_acts,
                episodes=12,
                match_seed_base=830000,
                run_label="acts-four",
                acts=4,
                temperature=1.0,
            )
            with self.assertRaisesRegex(ValueError, "mixed policyTemperature"):
                load_non_card_counterfactuals([str(first), str(second)])
            mixed = load_non_card_counterfactuals(
                [str(first), str(second)],
                allow_mixed_policy_temperatures=True,
                validation_fraction=0.3,
                split_seed=71,
            )
            self.assertEqual(len(mixed.files), 2)
            self.assertEqual(
                len(mixed.tax_return.train) + len(mixed.tax_return.validation),
                24,
            )
            with self.assertRaisesRegex(ValueError, "same acts horizon"):
                load_non_card_counterfactuals([str(first), str(different_acts)])

    def test_redundant_metadata_and_encoded_observation_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.ndjson"
            _write_fixture(source, episodes=4)
            source_lines = source.read_text(encoding="utf-8").splitlines()

            actor_seat_drift = root / "actor-seat-drift.ndjson"
            lines = list(source_lines)
            tax_record = json.loads(lines[1])
            tax_record["actorSeat"] = 0
            lines[1] = json.dumps(tax_record, separators=(",", ":"))
            actor_seat_drift.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "role features do not match"):
                load_non_card_counterfactuals([str(actor_seat_drift)])

            joker_drift = root / "joker-drift.ndjson"
            lines = list(source_lines)
            revolution_record = json.loads(lines[2])
            revolution_record["observation"][20] = 0.5
            revolution_record["observation"][8] = 1.0
            lines[2] = json.dumps(revolution_record, separators=(",", ":"))
            joker_drift.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both jokers"):
                load_non_card_counterfactuals([str(joker_drift)])

            action_feature_drift = root / "action-feature-drift.ndjson"
            lines = list(source_lines)
            revolution_record = json.loads(lines[2])
            revolution_record["actions"][1]["actionFeatures"] = [0.0, 1.0, 0.0]
            lines[2] = json.dumps(revolution_record, separators=(",", ":"))
            action_feature_drift.write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "action features mismatch"):
                load_non_card_counterfactuals([str(action_feature_drift)])

            decision_utility_drift = root / "decision-utility-drift.ndjson"
            lines = list(source_lines)
            tax_record = json.loads(lines[1])
            tax_record["actions"][0]["decisionActUtility"] = 0.0
            lines[1] = json.dumps(tax_record, separators=(",", ":"))
            decision_utility_drift.write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "decisionActUtility does not match"
            ):
                load_non_card_counterfactuals(
                    [str(decision_utility_drift)]
                )

    def test_temperature_override_and_exhaustive_losses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.ndjson"
            _write_fixture(path)
            datasets = load_non_card_counterfactuals(
                [str(path)], validation_fraction=0.3, split_seed=91
            )
            original = datasets.tax_return
            colder = apply_policy_temperature(original, 0.5)
            self.assertGreater(
                float(colder.train.policy_targets.max(axis=1).mean()),
                float(original.train.policy_targets.max(axis=1).mean()),
            )
            decision_act = apply_utility_target(
                original,
                utility_target="decision-act",
                temperature=1.0,
            )
            first_row = decision_act.train
            legal = first_row.legal_masks[0]
            raw = first_row.decision_act_utilities[0, legal]
            centered = first_row.action_value_targets[0, legal]
            self.assertTrue(
                np.allclose(centered, raw - raw.mean(), atol=1.0e-7)
            )
            expected_probabilities = np.exp(centered - centered.max())
            expected_probabilities /= expected_probabilities.sum()
            self.assertTrue(
                np.allclose(
                    first_row.policy_targets[0, legal],
                    expected_probabilities,
                    atol=1.0e-7,
                )
            )
            self.assertAlmostEqual(
                float(first_row.value_targets[0]),
                float(np.sum(expected_probabilities * raw)),
                places=6,
            )
            model = TaxReturnActionConditionedActorCriticNetwork(
                actor_observation_hidden_sizes=(8,),
                actor_action_hidden_sizes=(4,),
                actor_scorer_hidden_sizes=(),
                value_hidden_sizes=(8,),
            )
            arrays = colder.train
            observations = torch.from_numpy(arrays.observations[:4])
            masks = torch.from_numpy(arrays.legal_masks[:4])
            logits, values = model(observations, masks)
            unanchored_options = TrainingOptions(
                epochs=1,
                batch_size=4,
                behavior_cloning_coefficient=0.0,
            )
            loss, metrics = supervised_loss(
                logits,
                values,
                masks,
                torch.from_numpy(arrays.policy_targets[:4]),
                torch.from_numpy(arrays.action_value_targets[:4]),
                torch.from_numpy(arrays.action_weights[:4]),
                torch.from_numpy(arrays.value_targets[:4]),
                torch.from_numpy(arrays.best_actions[:4]),
                torch.from_numpy(arrays.baseline_actions[:4]),
                torch.from_numpy(arrays.sample_weights[:4]),
                unanchored_options,
            )
            anchored_loss, anchored_metrics = supervised_loss(
                logits,
                values,
                masks,
                torch.from_numpy(arrays.policy_targets[:4]),
                torch.from_numpy(arrays.action_value_targets[:4]),
                torch.from_numpy(arrays.action_weights[:4]),
                torch.from_numpy(arrays.value_targets[:4]),
                torch.from_numpy(arrays.best_actions[:4]),
                torch.from_numpy(arrays.baseline_actions[:4]),
                torch.from_numpy(arrays.sample_weights[:4]),
                TrainingOptions(
                    epochs=1,
                    batch_size=4,
                    behavior_cloning_coefficient=2.0,
                ),
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(metrics["policyKl"]), -1.0e-6)
            self.assertTrue(torch.isfinite(metrics["actorSelectionLoss"]))
            self.assertTrue(torch.isfinite(metrics["behaviorCloningLoss"]))
            self.assertTrue(
                torch.allclose(
                    anchored_metrics["actorSelectionLoss"],
                    metrics["actorSelectionLoss"]
                    + 2.0 * metrics["behaviorCloningLoss"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    anchored_loss,
                    loss + 2.0 * metrics["behaviorCloningLoss"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    metrics["totalLoss"],
                    metrics["actorSelectionLoss"]
                    + 0.25 * metrics["valueLoss"],
                )
            )
            anchored_loss.backward()
            self.assertIsNotNone(model.actor_scorer[-1].weight.grad)
            self.assertIsNotNone(model.value_network[-1].weight.grad)
            self.assertEqual(TrainingOptions().behavior_cloning_coefficient, 0.0)
            self.assertEqual(TrainingOptions().utility_target, "terminal")
            with self.assertRaisesRegex(ValueError, "behavior_cloning_coefficient"):
                validate_options(
                    TrainingOptions(behavior_cloning_coefficient=-0.1)
                )
            with self.assertRaisesRegex(ValueError, "utility_target"):
                validate_options(TrainingOptions(utility_target="invalid"))

    def test_cpu_training_verification_packaging_and_exclusive_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.ndjson"
            output = root / "training-run-new"
            archive = root / "training-run-new-result.zip"
            _write_fixture(data, episodes=24)
            options = TrainingOptions(
                epochs=3,
                batch_size=64,
                learning_rate=1.0e-3,
                early_stopping_patience=1,
                early_stopping_min_delta=1.0e6,
                validation_fraction=0.3,
                split_seed=42,
                seed=1234,
                policy_temperature=0.5,
                behavior_cloning_coefficient=1.0,
                utility_target="decision-act",
                device="cpu",
            )
            manifest = train_non_card_models(
                data_patterns=[str(data)],
                output_directory=output,
                decision="all",
                options=options,
            )
            self.assertEqual(manifest["groupSplitKey"], "canonicalWorldKey")
            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["behaviorCloningCoefficient"], 1.0)
            self.assertEqual(manifest["utilityTarget"], "decision-act")
            config = json.loads(
                (output / "training-config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["behaviorCloningCoefficient"], 1.0)
            self.assertEqual(config["utilityTarget"], "decision-act")
            self.assertEqual(
                config["options"]["behavior_cloning_coefficient"], 1.0
            )
            metrics = json.loads(
                (output / "training-metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metrics["decisions"]["tax-return"]["stoppedEarly"])
            self.assertTrue(metrics["decisions"]["revolution"]["stoppedEarly"])
            self.assertEqual(
                metrics["decisions"]["tax-return"]["selectionMetric"],
                "validation.actorSelectionLoss",
            )
            tax_validation = metrics["decisions"]["tax-return"]["history"][0][
                "validation"
            ]
            for metric_name in (
                "behaviorCloningLoss",
                "baselineActionAgreement",
                "targetBestEqualsBaselineRate",
                "predictedLogitMarginVsBaseline",
                "predictedProbabilityMarginVsBaseline",
                "targetUtilityMarginVsBaseline",
            ):
                self.assertIn(metric_name, tax_validation)
            checkpoint = torch.load(
                output / "tax-return" / "best" / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["version"], 3)
            self.assertEqual(checkpoint["behaviorCloningCoefficient"], 1.0)
            self.assertEqual(checkpoint["utilityTarget"], "decision-act")
            verification = verify_result_directory(output)
            self.assertEqual(
                verification["decisionKinds"], ["tax-return", "revolution"]
            )
            report = package_result_directory(output, archive)
            self.assertEqual(report["behaviorCloningCoefficient"], 1.0)
            self.assertEqual(report["packageVersion"], 3)
            self.assertEqual(report["utilityTarget"], "decision-act")
            self.assertTrue(Path(report["checksumFile"]).is_file())
            second_verification = verify_result_archive(archive)
            self.assertEqual(report["sha256"], second_verification["sha256"])
            replay_output = root / "training-run-reproducible"
            train_non_card_models(
                data_patterns=[str(data)],
                output_directory=replay_output,
                decision="tax-return",
                options=options,
            )
            self.assertEqual(
                hashlib.sha256(
                    (output / "tax-return" / "best" / "model.json").read_bytes()
                ).hexdigest(),
                hashlib.sha256(
                    (
                        replay_output
                        / "tax-return"
                        / "best"
                        / "model.json"
                    ).read_bytes()
                ).hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                train_non_card_models(
                    data_patterns=[str(data)],
                    output_directory=output,
                    decision="all",
                    options=options,
                )
            with self.assertRaises(FileExistsError):
                package_result_directory(output, archive)


if __name__ == "__main__":
    unittest.main()
