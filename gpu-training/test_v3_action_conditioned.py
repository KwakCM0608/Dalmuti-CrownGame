import json
import tempfile
import unittest
from pathlib import Path

import torch

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_COUNT,
    V3_ACTION_FEATURE_COUNT,
    V3_ACTION_FEATURES,
    V3ActionConditionedActorCriticNetwork,
    decode_v3_semantic_action,
    encode_v3_semantic_action,
    export_v3_action_conditioned_json,
    load_v3_action_conditioned_json,
)


class V3ActionConditionedTests(unittest.TestCase):
    def test_catalogue_contains_only_structurally_possible_actions(self) -> None:
        self.assertEqual(V3_ACTION_COUNT, 236)
        self.assertEqual(len(V3_ACTION_CATALOGUE), V3_ACTION_COUNT)
        self.assertEqual(len(V3_ACTION_FEATURES), V3_ACTION_COUNT)
        self.assertTrue(
            all(len(features) == V3_ACTION_FEATURE_COUNT for features in V3_ACTION_FEATURES)
        )
        for index, action in enumerate(V3_ACTION_CATALOGUE):
            self.assertEqual(encode_v3_semantic_action(action), index)
            self.assertEqual(decode_v3_semantic_action(index), action)
            if action["type"] == "play":
                natural_count = action["count"] - action["jokerCount"]
                self.assertGreaterEqual(natural_count, 1)
                self.assertLessEqual(natural_count, action["rank"])

    def test_network_scores_only_requested_legal_actions(self) -> None:
        torch.manual_seed(20260801)
        model = V3ActionConditionedActorCriticNetwork(
            observation_features=4,
            observation_schema_version=3,
            actor_observation_hidden_sizes=(3,),
            actor_action_hidden_sizes=(2,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(3,),
        ).eval()
        observations = torch.tensor(
            [[0.0, 0.25, 0.5, 1.0], [1.0, 0.5, 0.25, 0.0]],
            dtype=torch.float32,
        )
        legal = torch.tensor([0, 1, 235], dtype=torch.long)
        legal_logits, legal_values = model.forward_legal(observations, legal)
        all_logits, all_values = model(observations)
        legal_masks = torch.zeros((2, V3_ACTION_COUNT), dtype=torch.bool)
        legal_masks[:, legal] = True
        masked_logits, masked_values = model(observations, legal_masks)

        self.assertEqual(tuple(legal_logits.shape), (2, 3))
        self.assertEqual(tuple(legal_values.shape), (2,))
        self.assertTrue(torch.allclose(legal_logits, all_logits[:, legal]))
        self.assertTrue(torch.allclose(legal_values, all_values))
        self.assertTrue(torch.allclose(masked_logits[:, legal], legal_logits))
        self.assertTrue(torch.all(masked_logits[:, 2:235] == -1.0e9))
        self.assertTrue(torch.allclose(masked_values, legal_values))
        (masked_logits[legal_masks].sum() + masked_values.sum()).backward()
        self.assertIsNotNone(model.actor_scorer[-1].weight.grad)
        self.assertIsNotNone(model.value_network[-1].weight.grad)

    def test_json_export_and_import_preserve_outputs(self) -> None:
        torch.manual_seed(81)
        model = V3ActionConditionedActorCriticNetwork(
            observation_features=4,
            observation_schema_version=3,
            actor_observation_hidden_sizes=(3,),
            actor_action_hidden_sizes=(2,),
            actor_scorer_hidden_sizes=(4,),
            value_hidden_sizes=(3,),
        ).eval()
        observations = torch.tensor([[0.1, -0.2, 0.3, 0.4]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3-model.json"
            export_v3_action_conditioned_json(model, path)
            loaded, payload = load_v3_action_conditioned_json(path)
            original_logits, original_values = model(observations)
            loaded_logits, loaded_values = loaded.eval()(observations)

            self.assertEqual(
                payload["format"], "dalmuti-action-conditioned-actor-critic"
            )
            self.assertEqual(payload["actionCount"], 236)
            self.assertTrue(
                torch.allclose(original_logits, loaded_logits, atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(original_values, loaded_values, atol=1e-6)
            )

    def test_loader_rejects_action_contract_drift(self) -> None:
        model = V3ActionConditionedActorCriticNetwork(
            observation_features=2,
            actor_observation_hidden_sizes=(2,),
            actor_action_hidden_sizes=(2,),
            actor_scorer_hidden_sizes=(),
            value_hidden_sizes=(2,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3-model.json"
            export_v3_action_conditioned_json(model, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["actionCount"] = 235
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "catalogue contract"):
                load_v3_action_conditioned_json(path)


if __name__ == "__main__":
    unittest.main()
