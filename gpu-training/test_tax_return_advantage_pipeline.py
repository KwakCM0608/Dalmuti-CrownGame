from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from non_card_action_conditioned import (
    TAX_RETURN_ACTION_CATALOGUE,
    TAX_RETURN_ACTION_FEATURE_LAYOUT,
    TAX_RETURN_ACTION_FEATURES,
)
from tax_return_advantage import (
    BASELINE_PROVENANCE,
    BASELINE_PROVENANCE_SHA256,
    TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT,
    TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION,
    TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
    TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT,
    TaxReturnBilinearResidualNetwork,
    export_layer_parameters,
    member_parameters_sha256,
    validate_ensemble_payload,
)
from tax_return_advantage_dataset import (
    COUNTERFACTUAL_FORMAT,
    DETERMINIZATION_ALGORITHM,
    DETERMINIZATION_ALGORITHM_VERSION,
    DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
    DETERMINIZATION_CONTINUATION_RNG_PAIRING,
    DETERMINIZATION_CONTINUATION_SEED_DERIVATION,
    DETERMINIZATION_CONTRACT_SHA256,
    DETERMINIZATION_FORMAT_VERSION,
    DETERMINIZATION_SCHEMA,
    _load_v2,
    _validate_v2_manifest,
    _validate_v2_tax_record,
    canonical_information_state_key,
    simulator_reward_advantage_to_chips,
)
from package_tax_return_advantage_results import _verify_external_checksum
from train_tax_return_advantage import (
    TrainingOptions,
    group_bootstrap_indices,
    member_seed,
    paired_advantage_loss,
)
from verify_tax_return_advantage_results import _actual_payload_paths


class TaxReturnAdvantagePipelineTests(unittest.TestCase):
    @staticmethod
    def _v2_manifest() -> dict[str, object]:
        return {
            "type": "manifest",
            "format": COUNTERFACTUAL_FORMAT,
            "version": DETERMINIZATION_FORMAT_VERSION,
            "createdAt": "2026-08-01T00:00:00.000Z",
            "observationSchemaVersion": 1,
            "actionCatalogueVersions": {"taxReturn": 1, "revolution": 1},
            "featureDimensions": {
                "taxReturn": {"observation": 103, "action": 15, "catalogue": 103},
                "revolution": {"observation": 102, "action": 3, "catalogue": 2},
            },
            "collection": {
                "playerCounts": [4],
                "episodesPerPlayerCount": 2,
                "acts": 2,
                "initialSeed": 17,
                "matchSeedDerivation": (
                    "initialSeed + zero-based index over ascending playerCount then episode"
                ),
                "decisionKinds": ["tax-return"],
                "policyTemperature": 1.0,
                "maxDecisions": 2,
                "baselineNonCardHooks": {},
                "continuationPolicy": "normal-deterministic",
                "resumeAllowed": False,
                "taxReturnCounts": [2],
                "determinization": {
                    "worldCountPerInformationState": 2,
                    "continuationCountPerHiddenWorld": 2,
                    "rawContinuationEvaluationsPerInformationState": 4,
                    "effectiveIndependentWorldsPerInformationState": 2,
                    "standardErrorEstimable": True,
                    "originalReplayWorldIncluded": True,
                    "rootSeed": 23,
                    "maxAttemptsPerResampledWorld": 32,
                    "algorithm": DETERMINIZATION_ALGORITHM,
                    "algorithmVersion": DETERMINIZATION_ALGORITHM_VERSION,
                    "algorithmContractSha256": DETERMINIZATION_CONTRACT_SHA256,
                    "candidateSeedDerivation": DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
                    "continuationSeedDerivation": DETERMINIZATION_CONTINUATION_SEED_DERIVATION,
                },
            },
            "privacy": {
                "observation": "encoded-actor-hand-and-public-state-only",
                "opponentCardIdentitiesIncluded": False,
                "physicalCardIdsIncluded": False,
                "individualReplaySeedsIncluded": False,
                "explicitIndividualSeedsIncluded": False,
                "individualSeedsDerivableFromRestrictedRootProvenance": True,
                "individualWorldUtilitiesIncluded": False,
                "aggregateTargetsOnly": True,
                "distribution": "restricted-training-only",
            },
            "groupSplitKey": "canonicalInformationStateKey",
            "determinizationSchema": DETERMINIZATION_SCHEMA,
        }

    @staticmethod
    def _v2_record(predecision_sha: str = "0" * 64) -> dict[str, object]:
        observation = [0.0] * 103
        observation[0:3] = [0.0, 1 / 19, 2 / 20]
        observation[3] = 1.0
        observation[9] = 0.5
        observation[10] = 1 / 3
        roles = (0, 1, 3, 4)
        for slot, role_index in enumerate(roles):
            offset = 21 + slot * 8
            observation[offset] = 1.0
            observation[offset + 1] = 0.1 if slot == 0 else 0.5
            observation[offset + 3 + role_index] = 1.0
        observation[102] = 1.0
        action_index = TAX_RETURN_ACTION_CATALOGUE.index((2, 3))
        legal_mask = [index == action_index for index in range(103)]
        zero_stats = {
            "mean": 0.0,
            "sampleStandardDeviation": 0.0,
            "standardError": 0.0,
            "count": 2,
            "standardErrorEstimable": True,
        }
        zero_uncertainty = {
            key: value for key, value in zero_stats.items() if key != "mean"
        }
        record = {
            "type": "counterfactual-decision",
            "sampleId": f"tax-return:{predecision_sha}",
            "canonicalInformationStateKey": "",
            "decision": "tax-return",
            "playerCount": 4,
            "acts": 2,
            "round": 2,
            "actorId": "player-1",
            "actorSeat": 0,
            "actorRole": "great-dalmuti",
            "observationSchemaVersion": 1,
            "actionCatalogueVersion": 1,
            "observation": observation,
            "legalMask": legal_mask,
            "legalActionIndices": [action_index],
            "baselineActionIndex": action_index,
            "metadata": {"playerCount": 4, "actorHandCount": 2, "returnCount": 2},
            "pairing": {
                "canonicalInformationStateKey": "",
                "preDecisionSha256": predecision_sha,
                "continuationPolicy": "normal-deterministic",
                "forcedOverrideNamespace": "taxReturn",
                "rootActionCoverage": (
                    "all-legal-actions-in-every-accepted-hidden-world"
                ),
                "continuationRngPairing": DETERMINIZATION_CONTINUATION_RNG_PAIRING,
            },
            "determinization": {
                "worldCount": 2,
                "continuationCount": 2,
                "rawContinuationEvaluations": 4,
                "effectiveIndependentWorlds": 2,
                "standardErrorEstimable": True,
                "originalReplayWorldIncluded": True,
                "resampledWorldCount": 1,
                "rootSeed": 23,
                "maxAttemptsPerResampledWorld": 32,
                "algorithm": DETERMINIZATION_ALGORITHM,
                "algorithmVersion": DETERMINIZATION_ALGORITHM_VERSION,
                "algorithmContractSha256": DETERMINIZATION_CONTRACT_SHA256,
                "candidateSeedDerivation": DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
                "continuationSeedDerivation": DETERMINIZATION_CONTINUATION_SEED_DERIVATION,
                "acceptedWorldAttempts": [{
                    "worldIndex": 1,
                    "attemptCount": 1,
                    "rejectedAttemptCount": 0,
                    "rejectedReasonCounts": {},
                }],
                "individualReplaySeedsIncluded": False,
                "explicitIndividualSeedsIncluded": False,
                "individualSeedsDerivableFromRestrictedRootProvenance": True,
                "individualWorldUtilitiesIncluded": False,
                "distribution": "restricted-training-only",
            },
            "utility": {
                "terminalDefinition": "terminal-cumulative-chip-score",
                "decisionActDefinition": "centered-round-chip-award",
                "centeredAcrossLegalActions": True,
                "pairedBaselineAdvantagesBeforeAggregation": True,
            },
            "targetBuilder": (
                "training/non-card-search-targets.ts#buildPairedCounterfactualTargets"
            ),
            "targetSampleCount": 2,
            "bestActionIndex": action_index,
            "bestDecisionActActionIndex": action_index,
            "forcedActionEvaluations": 4,
            "actions": [{
                "actionIndex": action_index,
                "actionFeatures": list(TAX_RETURN_ACTION_FEATURES[action_index]),
                "meanUtility": 0.0,
                "centeredUtility": 0.0,
                "uncertainty": dict(zero_uncertainty),
                "softTargetProbability": 1.0,
                "pairedBaselineAdvantage": dict(zero_stats),
                "decisionActUtilityAggregate": {
                    "meanUtility": 0.0,
                    "centeredUtility": 0.0,
                    "uncertainty": dict(zero_uncertainty),
                    "softTargetProbability": 1.0,
                },
                "pairedDecisionActBaselineAdvantage": dict(zero_stats),
            }],
        }
        key = canonical_information_state_key(record)
        record["canonicalInformationStateKey"] = key
        record["pairing"]["canonicalInformationStateKey"] = key
        return record

    def test_baseline_residual_is_exact_zero(self) -> None:
        torch.manual_seed(7)
        model = TaxReturnBilinearResidualNetwork(context_features=3)
        observations = torch.randn(2, 103)
        masks = torch.zeros(2, 103, dtype=torch.bool)
        masks[0, [13, 14, 20]] = True
        masks[1, [40, 55]] = True
        baselines = torch.tensor([14, 55], dtype=torch.long)
        advantages = model(observations, baselines, masks)
        self.assertEqual(float(advantages[0, 14].detach()), 0.0)
        self.assertEqual(float(advantages[1, 55].detach()), 0.0)
        self.assertTrue(torch.equal(advantages[~masks], torch.zeros_like(advantages[~masks])))

    def test_paired_loss_weights_each_state_equally(self) -> None:
        predicted = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        targets = torch.zeros_like(predicted)
        legal_masks = torch.tensor(
            [[True, True, True, True], [True, True, False, False]],
            dtype=torch.bool,
        )
        baselines = torch.tensor([0, 0], dtype=torch.long)
        options = TrainingOptions(
            huber_delta_chips=1.0,
            regression_coefficient=1.0,
            sign_coefficient=0.0,
        )
        loss, metrics = paired_advantage_loss(
            predicted,
            targets,
            legal_masks,
            baselines,
            options,
        )
        # State 1 Huber values are .5, 1.5, 2.5 => mean 1.5.
        # State 2 has one comparison with value 3.5. Equal-state mean = 2.5.
        self.assertAlmostEqual(float(loss), 2.5, places=7)
        self.assertAlmostEqual(float(metrics["regression"]), 2.5, places=7)

    def test_bootstrap_and_member_seeds_are_deterministic(self) -> None:
        groups = ("a", "a", "b", "c")
        first = group_bootstrap_indices(groups, 1234)
        second = group_bootstrap_indices(groups, 1234)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, group_bootstrap_indices(groups, 1235)))
        seeds = [member_seed(77, index) for index in range(5)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seeds[0], 77)

    def test_v2_recomputes_canonical_state_key_and_rejects_semantic_corruption(self) -> None:
        manifest = self._v2_manifest()
        contract = _validate_v2_manifest(manifest, Path("fixture.ndjson"))
        record = self._v2_record()
        parsed = _validate_v2_tax_record(
            record,
            path=Path("fixture.ndjson"),
            line_number=2,
            manifest_contract=contract,
        )
        self.assertEqual(parsed["groupKey"], record["canonicalInformationStateKey"])

        fabricated_key = copy.deepcopy(record)
        fabricated_key["canonicalInformationStateKey"] = "sha256:" + "f" * 64
        fabricated_key["pairing"]["canonicalInformationStateKey"] = (
            fabricated_key["canonicalInformationStateKey"]
        )
        with self.assertRaisesRegex(ValueError, "does not match public state"):
            _validate_v2_tax_record(
                fabricated_key,
                path=Path("fixture.ndjson"),
                line_number=2,
                manifest_contract=contract,
            )

        corrupt_observation = copy.deepcopy(record)
        corrupt_observation["observation"][0] = 0.5
        corrupt_key = canonical_information_state_key(corrupt_observation)
        corrupt_observation["canonicalInformationStateKey"] = corrupt_key
        corrupt_observation["pairing"]["canonicalInformationStateKey"] = corrupt_key
        with self.assertRaisesRegex(ValueError, "global observation features"):
            _validate_v2_tax_record(
                corrupt_observation,
                path=Path("fixture.ndjson"),
                line_number=2,
                manifest_contract=contract,
            )

    def test_v2_binds_known_determinization_contract_and_rejects_duplicate_states(self) -> None:
        corrupt_manifest = self._v2_manifest()
        corrupt_manifest["collection"]["determinization"][
            "algorithmContractSha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "algorithm contract"):
            _validate_v2_manifest(corrupt_manifest, Path("fixture.ndjson"))

        manifest = self._v2_manifest()
        first = self._v2_record("1" * 64)
        second = self._v2_record("2" * 64)
        self.assertEqual(
            first["canonicalInformationStateKey"],
            second["canonicalInformationStateKey"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.ndjson"
            content_lines = [manifest, first, second]
            content = b"".join(
                (
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                for item in content_lines
            )
            summary = {
                "type": "summary",
                "hashes": {
                    "algorithm": "sha256",
                    "contentBeforeSummary": hashlib.sha256(content).hexdigest(),
                    "contentBeforeSummaryBytes": len(content),
                },
                "decisionsWritten": 2,
                "actionEvaluations": 8,
            }
            summary_bytes = (
                json.dumps(summary, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            path.write_bytes(content + summary_bytes)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".ndjson.sha256").write_text(
                f"{digest}  {path.name}\n",
                encoding="ascii",
                newline="\n",
            )
            with self.assertRaisesRegex(
                ValueError,
                "exactly one aggregate record per canonical information state",
            ):
                _load_v2([path], validation_fraction=0.5, split_seed=20260801)

    def test_reward_unit_transform_is_bound_to_actual_chips(self) -> None:
        self.assertEqual(simulator_reward_advantage_to_chips(0.5), 1.0)
        self.assertEqual(simulator_reward_advantage_to_chips(-1.0), -2.0)
        self.assertEqual(simulator_reward_advantage_to_chips(-0.0), 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            simulator_reward_advantage_to_chips(math.inf)

    def test_external_package_checksum_and_nested_inventory_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "result.zip"
            archive.write_bytes(b"archive bytes")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            checksum = root / "result.zip.sha256"
            checksum.write_text(
                f"{digest}  {archive.name}\n",
                encoding="ascii",
                newline="\n",
            )
            self.assertEqual(_verify_external_checksum(archive, checksum), digest)
            checksum.write_text(
                f"{'f' * 64}  {archive.name}\n",
                encoding="ascii",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "external checksum mismatch"):
                _verify_external_checksum(archive, checksum)

            (root / "training-manifest.json").write_text("root", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "training-manifest.json").write_text(
                "must-not-be-ignored", encoding="utf-8"
            )
            self.assertIn(
                "nested/training-manifest.json",
                _actual_payload_paths(root),
            )

    def test_strict_v2_ensemble_round_trips_five_hashed_members(self) -> None:
        members = []
        for index in range(5):
            model = TaxReturnBilinearResidualNetwork(context_features=2)
            parameters = export_layer_parameters(model)
            member = {
                "memberIndex": index,
                "seed": member_seed(5, index),
                "checkpointEpoch": index + 1,
                "validationPairedLoss": 0.1 + index * 0.01,
                "parametersSha256": "",
                **parameters,
            }
            member["parametersSha256"] = member_parameters_sha256(member)
            members.append(member)
        payload = {
            "format": TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT,
            "version": TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION,
            "decisionKind": "tax-return",
            "scoreSemantics": TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
            "observationSchemaVersion": 1,
            "observationFeatures": 103,
            "actionCatalogueVersion": 1,
            "actionCount": 103,
            "actionFeatures": 15,
            "actionFeatureLayout": list(TAX_RETURN_ACTION_FEATURE_LAYOUT),
            "trainingData": {
                "sourceFormatVersions": [2],
                "groupSplitKey": "canonicalInformationStateKey",
                "determinizationSchema": "world-clustered-paired-baseline-advantages-v2",
                "worldCountPerInformationState": 8,
                "continuationCountPerHiddenWorld": 4,
                "effectiveIndependentWorldsPerInformationState": 8,
                "rawContinuationEvaluationsPerInformationState": 32,
                "standardErrorEstimable": True,
                "determinizationAlgorithm": "target-act-opponent-physical-card-fisher-yates-v1",
                "determinizationAlgorithmVersion": 1,
                "determinizationAlgorithmContractSha256": "368240f14f2e5d84bb3085610a176ad4519bc6e5ae288b70de549f63212905c4",
                "candidateSeedDerivation": "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,attempt)))",
                "continuationSeedDerivation": "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,continuationIndex,continuation)))",
                "targetField": "actions[].pairedDecisionActBaselineAdvantage.mean",
                "targetTransform": {
                    "scoreUnit": "chip-units",
                    "sourceUnit": "(roundChipAward-2)/2",
                    "operation": "multiply-source-baseline-advantage-by-2",
                    "multiplier": 2.0,
                },
                "stateWeighting": "one-per-information-state-independent-of-worldCount",
            },
            "architecture": {
                "contextFeatures": 2,
                "contextActivation": "tanh",
                "score": "raw(s,a)-raw(s,normalBaselineAction)",
                "weightLayout": TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT,
            },
            "baseline": {
                "provenance": BASELINE_PROVENANCE,
                "provenanceSha256": BASELINE_PROVENANCE_SHA256,
                "score": "exactly-zero-by-residualization",
            },
            "objective": {
                "utilityTarget": "decision-act-current-chip-advantage",
                "utilityScale": "chip-units",
                "weighting": "equal-per-state",
                "regression": {
                    "loss": "huber-paired-action-vs-baseline",
                    "coefficient": 1.0,
                    "deltaChips": 0.5,
                },
                "tieAwareSign": {
                    "loss": "binary-cross-entropy-with-logits",
                    "coefficient": 0.25,
                    "temperatureChips": 0.25,
                    "tieTarget": 0.5,
                    "tieEpsilonChips": 1.0e-9,
                },
                "checkpointSelection": "paired-validation-loss",
                "bootstrapUnit": "canonicalInformationStateKey",
            },
            "routing": {
                "returnCountOne": "exact-normal-fallback",
                "returnCountTwo": "ensemble-lower-confidence-bound",
                "roleRouting": {
                    "great-dalmuti": "ensemble-lower-confidence-bound",
                    "lesser-dalmuti": "exact-normal-fallback",
                    "other-roles": "not-applicable",
                },
                "memberCount": 5,
                "unanimityRule": "all-member-advantages-strictly-positive",
                "lowerConfidenceBound": "mean-minus-z-times-sample-sd",
                "zValue": 1.645,
                "defaultMinimumChipAdvantage": 0.5,
                "selection": "maximum-eligible-lcb",
                "tieBreak": "baseline-then-lowest-action-index",
            },
            "members": members,
        }
        self.assertIs(validate_ensemble_payload(payload), payload)
        v1_payload = copy.deepcopy(payload)
        v1_payload["trainingData"].update(
            {
                "sourceFormatVersions": [1],
                "groupSplitKey": "canonicalWorldKey",
                "determinizationSchema": None,
                "worldCountPerInformationState": 1,
                "continuationCountPerHiddenWorld": 1,
                "effectiveIndependentWorldsPerInformationState": 1,
                "rawContinuationEvaluationsPerInformationState": 1,
                "standardErrorEstimable": False,
                "determinizationAlgorithm": None,
                "determinizationAlgorithmVersion": None,
                "determinizationAlgorithmContractSha256": None,
                "candidateSeedDerivation": None,
                "continuationSeedDerivation": None,
                "targetField": (
                    "actions[].decisionActUtility-minus-baseline.decisionActUtility"
                ),
            }
        )
        v1_payload["objective"]["bootstrapUnit"] = "canonicalWorldKey"
        self.assertIs(validate_ensemble_payload(v1_payload), v1_payload)
        payload["members"][0]["bilinearWeight"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "parameter hash mismatch"):
            validate_ensemble_payload(payload)


if __name__ == "__main__":
    unittest.main()
