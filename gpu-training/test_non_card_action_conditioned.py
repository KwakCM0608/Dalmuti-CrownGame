import json
import tempfile
import unittest
from pathlib import Path

import torch

from non_card_action_conditioned import (
    GREAT_PEON_ROLE_FEATURE_INDEX,
    MASKED_LOGIT,
    REVOLUTION_ACTION_COUNT,
    REVOLUTION_ACTION_FEATURE_COUNT,
    REVOLUTION_OBSERVATION_FEATURE_COUNT,
    TAX_RETURN_ACTION_CATALOGUE,
    TAX_RETURN_ACTION_COUNT,
    TAX_RETURN_ACTION_FEATURE_COUNT,
    TAX_RETURN_ACTION_FEATURES,
    TAX_RETURN_OBSERVATION_FEATURE_COUNT,
    RevolutionActionConditionedActorCriticNetwork,
    TaxReturnActionConditionedActorCriticNetwork,
    export_revolution_action_conditioned_json,
    export_tax_return_action_conditioned_json,
    legal_revolution_masks_from_observations,
    legal_tax_return_masks_from_observations,
    load_revolution_action_conditioned_json,
    load_tax_return_action_conditioned_json,
    revolution_action_features_from_observations,
)


def tax_observation(return_count: int = 2) -> torch.Tensor:
    observation = torch.zeros(TAX_RETURN_OBSERVATION_FEATURE_COUNT)
    observation[0] = 2.0 / 6.0  # six players
    observation[1] = 1.0 / 19.0  # second act
    observation[2] = 4.0 / 20.0  # four private cards
    observation[3 if return_count == 2 else 4] = 1.0
    # Physical counts: rank 1 x1, rank 2 x2, joker x1.
    observation[8] = 1.0
    observation[9] = 1.0
    observation[20] = 0.5
    observation[101 if return_count == 1 else 102] = 1.0
    return observation


def revolution_observation(*, great_peon: bool) -> torch.Tensor:
    observation = torch.zeros(REVOLUTION_OBSERVATION_FEATURE_COUNT)
    observation[0] = 2.0 / 6.0
    observation[1] = 1.0 / 19.0
    observation[2] = 3.0 / 20.0
    observation[GREAT_PEON_ROLE_FEATURE_INDEX if great_peon else 3] = 1.0
    observation[20] = 1.0  # both jokers
    observation[101] = 1.0  # taxation applies
    return observation


class NonCardActionConditionedTests(unittest.TestCase):
    def test_python_contract_matches_typescript_feature_sizes(self) -> None:
        self.assertEqual(TAX_RETURN_OBSERVATION_FEATURE_COUNT, 103)
        self.assertEqual(REVOLUTION_OBSERVATION_FEATURE_COUNT, 102)
        self.assertEqual(TAX_RETURN_ACTION_COUNT, 103)
        self.assertEqual(TAX_RETURN_ACTION_FEATURE_COUNT, 15)
        self.assertEqual(REVOLUTION_ACTION_COUNT, 2)
        self.assertEqual(REVOLUTION_ACTION_FEATURE_COUNT, 3)
        self.assertEqual(TAX_RETURN_ACTION_CATALOGUE[:3], ((1,), (2,), (3,)))
        self.assertEqual(TAX_RETURN_ACTION_CATALOGUE[13], (1, 2))
        self.assertNotIn((1, 1), TAX_RETURN_ACTION_CATALOGUE)
        self.assertEqual(len(TAX_RETURN_ACTION_FEATURES), 103)
        self.assertTrue(
            all(len(features) == 15 for features in TAX_RETURN_ACTION_FEATURES)
        )

    def test_tax_network_derives_mask_and_scores_only_legal_actions(self) -> None:
        torch.manual_seed(20260801)
        observations = torch.stack((tax_observation(), tax_observation(1)))
        masks = legal_tax_return_masks_from_observations(observations)
        expected_two_card_actions = {(1, 2), (1, 13), (2, 2), (2, 13)}
        actual_two_card_actions = {
            TAX_RETURN_ACTION_CATALOGUE[index]
            for index in masks[0].nonzero(as_tuple=True)[0].tolist()
        }
        self.assertEqual(actual_two_card_actions, expected_two_card_actions)
        self.assertEqual(int(masks[1].sum()), 3)

        model = TaxReturnActionConditionedActorCriticNetwork(
            actor_observation_hidden_sizes=(8,),
            actor_action_hidden_sizes=(4,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(8,),
        )
        logits, values = model(observations, masks)
        self.assertEqual(tuple(logits.shape), (2, 103))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(torch.all(logits[~masks] == MASKED_LOGIT))
        self.assertTrue(torch.isfinite(logits[masks]).all())

        loss = logits[masks].sum() + values.sum()
        loss.backward()
        self.assertIsNotNone(model.actor_observation_trunk[0].weight.grad)
        self.assertIsNotNone(model.actor_action_trunk[0].weight.grad)
        self.assertIsNotNone(model.actor_scorer[-1].weight.grad)
        self.assertIsNotNone(model.value_network[-1].weight.grad)

        wrong_mask = masks.clone()
        wrong_mask[0, masks[0].nonzero(as_tuple=True)[0][0]] = False
        with self.assertRaisesRegex(ValueError, "does not match"):
            model(observations, wrong_mask)

    def test_tax_mask_rejects_nonphysical_encoded_counts(self) -> None:
        observation = tax_observation().unsqueeze(0)
        observation[0, 8] = 0.4
        with self.assertRaisesRegex(ValueError, "valid card counts"):
            legal_tax_return_masks_from_observations(observation)

    def test_revolution_features_masks_and_gradients(self) -> None:
        observations = torch.stack(
            (
                revolution_observation(great_peon=False),
                revolution_observation(great_peon=True),
            )
        )
        features = revolution_action_features_from_observations(observations)
        self.assertEqual(features[0, 0].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(features[0, 1].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(features[1, 1].tolist(), [0.0, 0.0, 1.0])
        masks = legal_revolution_masks_from_observations(observations)
        self.assertTrue(masks.all())

        model = RevolutionActionConditionedActorCriticNetwork(
            actor_observation_hidden_sizes=(8,),
            actor_action_hidden_sizes=(4,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(8,),
        )
        logits, values = model(observations, masks)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        (logits.sum() + values.sum()).backward()
        self.assertIsNotNone(model.actor_observation_trunk[0].weight.grad)
        self.assertIsNotNone(model.actor_action_trunk[0].weight.grad)
        self.assertIsNotNone(model.actor_scorer[-1].weight.grad)
        self.assertIsNotNone(model.value_network[-1].weight.grad)

        invalid_mask = masks.clone()
        invalid_mask[0, 1] = False
        with self.assertRaisesRegex(ValueError, "decline and declare"):
            model(observations, invalid_mask)

    def test_separate_json_roundtrips_preserve_outputs(self) -> None:
        torch.manual_seed(81)
        tax_model = TaxReturnActionConditionedActorCriticNetwork(
            actor_observation_hidden_sizes=(7,),
            actor_action_hidden_sizes=(5,),
            actor_scorer_hidden_sizes=(6,),
            value_hidden_sizes=(7,),
        ).eval()
        revolution_model = RevolutionActionConditionedActorCriticNetwork(
            actor_observation_hidden_sizes=(7,),
            actor_action_hidden_sizes=(5,),
            actor_scorer_hidden_sizes=(6,),
            value_hidden_sizes=(7,),
        ).eval()
        tax_input = tax_observation().unsqueeze(0)
        revolution_input = revolution_observation(
            great_peon=True
        ).unsqueeze(0)

        with tempfile.TemporaryDirectory() as directory:
            tax_path = Path(directory) / "tax-return-model.json"
            revolution_path = Path(directory) / "revolution-model.json"
            export_tax_return_action_conditioned_json(tax_model, tax_path)
            export_revolution_action_conditioned_json(
                revolution_model, revolution_path
            )
            loaded_tax, tax_payload = load_tax_return_action_conditioned_json(
                tax_path
            )
            loaded_revolution, revolution_payload = (
                load_revolution_action_conditioned_json(revolution_path)
            )

            original_tax = tax_model(tax_input)
            roundtrip_tax = loaded_tax.eval()(tax_input)
            original_revolution = revolution_model(revolution_input)
            roundtrip_revolution = loaded_revolution.eval()(
                revolution_input
            )
            self.assertEqual(tax_payload["decisionKind"], "tax-return")
            self.assertEqual(
                revolution_payload["decisionKind"], "revolution"
            )
            for original, roundtrip in zip(original_tax, roundtrip_tax):
                self.assertTrue(torch.allclose(original, roundtrip, atol=1e-6))
            for original, roundtrip in zip(
                original_revolution, roundtrip_revolution
            ):
                self.assertTrue(torch.allclose(original, roundtrip, atol=1e-6))

    def test_separate_loaders_reject_schema_drift(self) -> None:
        tax_model = TaxReturnActionConditionedActorCriticNetwork(
            actor_observation_hidden_sizes=(4,),
            actor_action_hidden_sizes=(3,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(4,),
        )
        revolution_model = RevolutionActionConditionedActorCriticNetwork(
            actor_observation_hidden_sizes=(4,),
            actor_action_hidden_sizes=(3,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(4,),
        )
        with tempfile.TemporaryDirectory() as directory:
            tax_path = Path(directory) / "tax.json"
            revolution_path = Path(directory) / "revolution.json"
            export_tax_return_action_conditioned_json(tax_model, tax_path)
            export_revolution_action_conditioned_json(
                revolution_model, revolution_path
            )

            tax_payload = json.loads(tax_path.read_text(encoding="utf-8"))
            tax_payload["observationFeatures"] = 102
            tax_path.write_text(json.dumps(tax_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observation contract"):
                load_tax_return_action_conditioned_json(tax_path)

            revolution_payload = json.loads(
                revolution_path.read_text(encoding="utf-8")
            )
            revolution_payload["greatPeonRoleFeatureIndex"] = 6
            revolution_path.write_text(
                json.dumps(revolution_payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "role-conditioned"):
                load_revolution_action_conditioned_json(revolution_path)


if __name__ == "__main__":
    unittest.main()
