import unittest

import torch

from v4_model import (
    V4_ACTION_COUNT,
    V4_MASKED_LOGIT,
    V4ActorConfig,
    V4CenteredLogitEnsemble,
    V4CriticConfig,
    V4PrivilegedQCritic,
    V4PublicActor,
    assert_actor_critic_parameter_isolation,
    centered_legal_logits,
)


def tiny_actor_config() -> V4ActorConfig:
    return V4ActorConfig(
        max_players=4,
        max_history=3,
        d_model=24,
        layers=1,
        heads=4,
        feedforward=48,
        action_hidden=16,
    )


def public_inputs(config: V4ActorConfig, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(73)
    global_features = torch.randn(
        batch_size, config.global_features, generator=generator
    )
    rank_features = torch.randn(
        batch_size, config.rank_tokens, config.rank_features, generator=generator
    )
    player_features = torch.randn(
        batch_size, config.max_players, config.player_features, generator=generator
    )
    player_mask = torch.ones(batch_size, config.max_players, dtype=torch.bool)
    memory_trace_features = torch.randn(
        batch_size,
        config.memory_tokens,
        config.memory_features,
        generator=generator,
    )
    history_features = torch.randn(
        batch_size, config.max_history, config.history_features, generator=generator
    )
    history_mask = torch.tensor([[True, True, False]] * batch_size)
    legal_masks = torch.zeros(batch_size, V4_ACTION_COUNT, dtype=torch.bool)
    legal_masks[:, [0, 1, 12, 80, 235]] = True
    return (
        global_features,
        rank_features,
        player_features,
        player_mask,
        memory_trace_features,
        history_features,
        history_mask,
        legal_masks,
    )


class V4ModelTests(unittest.TestCase):
    def test_production_defaults_match_v4_contract(self) -> None:
        config = V4ActorConfig()
        self.assertEqual(config.d_model, 384)
        self.assertEqual(config.layers, 8)
        self.assertEqual(config.heads, 12)
        self.assertEqual(config.max_history, 192)
        self.assertEqual(config.memory_tokens, 4)
        self.assertEqual(config.memory_features, 20)
        self.assertEqual(config.history_features, 20)

    def test_actor_is_public_only_and_invariant_to_hidden_state(self) -> None:
        torch.manual_seed(101)
        actor_config = tiny_actor_config()
        critic_config = V4CriticConfig(
            privileged_features=18,
            d_model=24,
            hidden_layers=1,
            action_hidden=16,
        )
        actor = V4PublicActor(actor_config).eval()
        critic = V4PrivilegedQCritic(critic_config).eval()
        assert_actor_critic_parameter_isolation(actor, critic)
        inputs = public_inputs(actor_config)
        before = actor(*inputs)
        hidden_a = torch.zeros(2, critic_config.privileged_features)
        hidden_b = torch.randn(2, critic_config.privileged_features)
        q_a = critic(hidden_a, inputs[-1])
        q_b = critic(hidden_b, inputs[-1])
        after = actor(*inputs)

        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.allclose(q_a[:, [0, 1]], q_b[:, [0, 1]]))
        self.assertFalse(
            any("privileged" in name.lower() for name, _ in actor.named_parameters())
        )
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        critic_ids = {id(parameter) for parameter in critic.parameters()}
        self.assertTrue(actor_ids.isdisjoint(critic_ids))

    def test_actor_and_critic_mask_illegal_actions_and_remain_finite(self) -> None:
        torch.manual_seed(103)
        actor_config = tiny_actor_config()
        critic_config = V4CriticConfig(
            privileged_features=18,
            d_model=24,
            hidden_layers=1,
            action_hidden=16,
        )
        actor = V4PublicActor(actor_config)
        critic = V4PrivilegedQCritic(critic_config)
        inputs = public_inputs(actor_config)
        logits = actor(*inputs)
        q_values = critic(
            torch.randn(2, critic_config.privileged_features), inputs[-1]
        )
        legal = inputs[-1]

        self.assertEqual(tuple(logits.shape), (2, 236))
        self.assertEqual(tuple(q_values.shape), (2, 236))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(q_values).all())
        self.assertTrue(torch.all(logits[~legal] == V4_MASKED_LOGIT))
        self.assertTrue(torch.all(q_values[~legal] == V4_MASKED_LOGIT))
        (logits[legal].sum() + q_values[legal].sum()).backward()
        self.assertIsNotNone(actor.action_bias.weight.grad)
        self.assertIsNotNone(critic.state_query[-1].weight.grad)
        self.assertIsNotNone(critic.action_bias.weight.grad)

    def test_three_seed_centered_logit_ensemble_matches_manual_average(self) -> None:
        config = tiny_actor_config()
        ensemble = V4CenteredLogitEnsemble.from_seeds(config, (11, 13, 17)).eval()
        inputs = public_inputs(config)
        actual = ensemble(*inputs)
        manual = torch.stack(
            [centered_legal_logits(actor(*inputs), inputs[-1]) for actor in ensemble.actors]
        ).mean(dim=0)
        manual = manual.masked_fill(~inputs[-1], V4_MASKED_LOGIT)

        self.assertEqual(ensemble.seeds, (11, 13, 17))
        self.assertTrue(torch.allclose(actual, manual, atol=1e-6))
        self.assertTrue(torch.isfinite(actual).all())
        legal_mean = (
            actual.masked_fill(~inputs[-1], 0.0).sum(dim=-1)
            / inputs[-1].sum(dim=-1)
        )
        self.assertTrue(torch.allclose(legal_mean, torch.zeros_like(legal_mean), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
