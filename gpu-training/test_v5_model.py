import inspect
import math
import unittest

import torch
from torch.nn import functional as F

from v5_contract import (
    V5_DECK_COUNTS,
    V5_MAX_HISTORY,
    V5_MAX_OPPONENTS,
    V5_MAX_PLAYERS,
    V5_PUBLIC_SCHEMA_VERSION,
    V5_RANK_COUNT,
)
from v5_model import (
    V5_ACTION_COUNT,
    V5_MASKED_LOGIT,
    V5ActorConfig,
    V5CentralStateValueCritic,
    V5CriticConfig,
    V5PublicActor,
    assert_actor_critic_parameter_isolation,
    canonical_v5_policy_numerics_contract,
    configure_v5_policy_numerics,
    normal_action_auxiliary_loss,
    normal_prior_logits,
    pack_legal_actions,
    trainable_parameter_count,
    validate_v5_policy_numerics_contract,
)
from v5_public import V5ActorPublicBatch


def tiny_actor_config() -> V5ActorConfig:
    return V5ActorConfig(
        history_latents=2,
        d_model=32,
        core_layers=1,
        action_layers=2,
        heads=4,
        feedforward=64,
    )


def public_batch(*, batch_size: int = 3) -> tuple[V5ActorPublicBatch, torch.Tensor]:
    generator = torch.Generator().manual_seed(5501)
    global_codes = torch.tensor(
        [V5_PUBLIC_SCHEMA_VERSION, 4, 1, 0, 0, 0], dtype=torch.long
    ).repeat(batch_size, 1)
    own = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long
    ).repeat(batch_size, 1)
    played = torch.zeros(batch_size, V5_RANK_COUNT, dtype=torch.long)
    deck = torch.tensor(V5_DECK_COUNTS, dtype=torch.long).unsqueeze(0)
    unknown = deck - own - played

    player_codes = torch.zeros(batch_size, V5_MAX_PLAYERS, 6, dtype=torch.long)
    player_codes[:, :4, 0] = torch.arange(4)
    player_codes[:, :4, 1] = torch.tensor([13, 20, 20, 14])
    player_codes[:, :4, 2] = torch.tensor([0, 1, 3, 4])
    player_mask = torch.zeros(batch_size, V5_MAX_PLAYERS, dtype=torch.bool)
    player_mask[:, :4] = True

    history_codes = torch.zeros(
        batch_size, V5_MAX_HISTORY, 12, dtype=torch.long
    )
    history_mask = torch.zeros(batch_size, V5_MAX_HISTORY, dtype=torch.bool)
    # One valid categorical play event; the rest is canonical padding.
    history_codes[:, 0, [0, 2, 3, 4, 5, 7]] = torch.tensor(
        [1, 13, 12, 1, 1, 1]
    )
    history_mask[:, 0] = True

    opponent_mask = torch.zeros(batch_size, V5_MAX_OPPONENTS, dtype=torch.bool)
    opponent_mask[:, :3] = True
    expected = torch.zeros(
        batch_size, V5_MAX_OPPONENTS, V5_RANK_COUNT
    )
    probability_one = torch.zeros_like(expected)
    probability_required = torch.zeros_like(expected)
    response = torch.zeros(batch_size, V5_MAX_OPPONENTS)
    expected[:, :3] = torch.rand(
        batch_size, 3, V5_RANK_COUNT, generator=generator
    )
    probability_one[:, :3] = torch.rand(
        batch_size, 3, V5_RANK_COUNT, generator=generator
    )
    probability_required[:, :3] = torch.rand(
        batch_size, 3, V5_RANK_COUNT, generator=generator
    )
    response[:, :3] = torch.rand(batch_size, 3, generator=generator)

    legal = torch.zeros(batch_size, V5_ACTION_COUNT, dtype=torch.bool)
    normal = torch.empty(batch_size, dtype=torch.long)
    choices = (
        (0, 5, 17, 80, 235),
        (1,),
        (0, 2, 4, 6, 8, 10, 12),
    )
    normals = (17, -1, 8)
    for row in range(batch_size):
        legal[row, list(choices[row % len(choices)])] = True
        normal[row] = normals[row % len(normals)]

    return V5ActorPublicBatch(
        global_codes=global_codes,
        own_rank_counts=own,
        public_played_counts=played,
        player_codes=player_codes,
        player_mask=player_mask,
        table_codes=torch.zeros(batch_size, 6, dtype=torch.long),
        history_codes=history_codes,
        history_mask=history_mask,
        legal_mask=legal,
        belief_unknown_rank_counts=unknown,
        belief_expected_counts=expected,
        belief_probability_at_least_one=probability_one,
        belief_probability_at_least_required=probability_required,
        belief_response_feasibility=response,
        opponent_mask=opponent_mask,
    ), normal


class V5ModelTests(unittest.TestCase):
    def test_policy_numerics_contract_is_canonical_and_applied(self) -> None:
        expected = canonical_v5_policy_numerics_contract()
        self.assertEqual(configure_v5_policy_numerics("cpu"), expected)
        self.assertEqual(validate_v5_policy_numerics_contract(expected), expected)
        changed = dict(expected)
        changed["mathSdp"] = False
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            validate_v5_policy_numerics_contract(changed)

    def test_production_actor_is_near_nine_million_parameters(self) -> None:
        actor = V5PublicActor()
        count = trainable_parameter_count(actor)

        self.assertGreaterEqual(count, 8_000_000)
        self.assertLessEqual(count, 9_000_000)
        self.assertEqual(actor.config.action_layers, 2)
        self.assertEqual(actor.config.history_latents, 8)

    def test_initial_policy_is_exact_normal_greedy_and_ninety_percent(self) -> None:
        torch.manual_seed(5511)
        actor = V5PublicActor(tiny_actor_config()).eval()
        batch, normal = public_batch()
        output = actor.forward_with_auxiliary(batch, normal)

        expected = torch.tensor([17, 1, 8])
        self.assertTrue(torch.equal(output.normal_actions, expected))
        self.assertTrue(torch.equal(output.logits.argmax(dim=-1), expected))
        self.assertTrue(
            torch.equal(
                output.residual_logits[batch.legal_mask],
                torch.zeros(13),
            )
        )
        probabilities = output.logits.softmax(dim=-1)
        self.assertTrue(
            torch.allclose(
                probabilities[[0, 2], expected[[0, 2]]],
                torch.tensor([0.9, 0.9]),
                atol=1e-6,
            )
        )
        self.assertEqual(float(probabilities[1, 1].detach()), 1.0)
        self.assertTrue(
            torch.all(output.logits[~batch.legal_mask] == V5_MASKED_LOGIT)
        )

    def test_prior_margin_is_log_nine_times_other_legal_actions(self) -> None:
        legal = torch.zeros(2, V5_ACTION_COUNT, dtype=torch.bool)
        legal[0, [2, 3]] = True
        legal[1, [10, 11, 12, 13, 14, 15]] = True
        normal = torch.tensor([3, 12])
        logits, resolved = normal_prior_logits(legal, normal)

        self.assertTrue(torch.equal(resolved, normal))
        self.assertAlmostEqual(float(logits[0, 3]), math.log(9.0), places=6)
        self.assertAlmostEqual(float(logits[1, 12]), math.log(45.0), places=6)
        probabilities = logits.softmax(dim=-1)
        self.assertTrue(
            torch.allclose(
                probabilities.gather(1, normal[:, None]).squeeze(1),
                torch.full((2,), 0.9),
            )
        )

    def test_actor_consumes_only_the_public_belief_batch(self) -> None:
        actor = V5PublicActor(tiny_actor_config())
        parameters = inspect.signature(actor.forward).parameters
        self.assertEqual(tuple(parameters), ("batch", "normal_actions"))
        self.assertFalse(
            any("privileged" in name.lower() for name, _ in actor.named_parameters())
        )
        batch, normal = public_batch()
        output = actor.forward_with_auxiliary(batch, normal)
        loss = normal_action_auxiliary_loss(
            output.normal_auxiliary_logits, output.normal_actions
        )
        loss.backward()
        gradient = actor.normal_auxiliary_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_hypergeometric_beliefs_are_real_actor_features(self) -> None:
        torch.manual_seed(5517)
        actor = V5PublicActor(tiny_actor_config()).eval()
        batch, normal = public_batch(batch_size=1)
        with torch.no_grad():
            actor.residual_output.weight.normal_(0.0, 0.02)
        before = actor(batch, normal)
        modified_values = {
            field: getattr(batch, field)
            for field in batch.__dataclass_fields__
        }
        modified_values["belief_probability_at_least_one"] = (
            batch.belief_probability_at_least_one.clone()
        )
        modified_values["belief_probability_at_least_one"][0, 0, 0] = 1.0
        after = actor(V5ActorPublicBatch(**modified_values), normal)

        self.assertFalse(torch.equal(before, after))

    def test_residual_training_can_overtake_normal(self) -> None:
        torch.manual_seed(5521)
        actor = V5PublicActor(tiny_actor_config())
        batch, normal = public_batch(batch_size=1)
        alternative = torch.tensor([235])
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.03)

        self.assertEqual(int(actor(batch, normal).argmax(dim=-1)), 17)
        for _ in range(35):
            optimizer.zero_grad(set_to_none=True)
            logits = actor(batch, normal)
            F.cross_entropy(logits, alternative).backward()
            optimizer.step()
        trained_logits = actor(batch, normal)

        self.assertEqual(int(trained_logits.argmax(dim=-1)), 235)
        self.assertGreater(
            float(trained_logits[0, 235].detach()),
            float(trained_logits[0, 17].detach()),
        )

    def test_dense_and_packed_legal_logits_and_actions_match(self) -> None:
        torch.manual_seed(5531)
        actor = V5PublicActor(tiny_actor_config()).eval()
        batch, normal = public_batch()
        with torch.no_grad():
            actor.residual_output.weight.normal_(0.0, 0.02)
            actor.residual_output.bias.fill_(0.01)
        dense = actor(batch, normal)
        packed_indices, packed_mask = pack_legal_actions(batch.legal_mask)
        packed = actor.forward_packed_batch(
            batch, normal, packed_indices, packed_mask
        )
        dense_selected = dense.gather(1, packed_indices).masked_fill(
            ~packed_mask, V5_MASKED_LOGIT
        )

        self.assertTrue(
            torch.allclose(packed.logits, dense_selected, atol=2e-5, rtol=2e-5)
        )
        self.assertTrue(
            torch.equal(packed.greedy_actions(), dense.argmax(dim=-1))
        )

    def test_amp_dense_and_packed_keep_fp32_prior_logits_and_gradient_parity(self) -> None:
        cases = [("cpu", torch.bfloat16)]
        if torch.cuda.is_available():
            cases.append(("cuda", torch.float16))
        for device_name, amp_dtype in cases:
            with self.subTest(device=device_name, dtype=str(amp_dtype)):
                torch.manual_seed(5532)
                actor = V5PublicActor(tiny_actor_config()).to(device_name).train()
                batch, normal = public_batch()
                batch = V5ActorPublicBatch(**{
                    field: getattr(batch, field).to(device_name)
                    for field in batch.__dataclass_fields__
                })
                normal = normal.to(device_name)
                packed_indices, packed_mask = pack_legal_actions(batch.legal_mask)

                with torch.amp.autocast(device_name, dtype=amp_dtype):
                    dense_output = actor.forward_with_auxiliary(batch, normal)
                    dense_selected = dense_output.logits.gather(
                        1, packed_indices
                    )[packed_mask]
                    dense_loss = dense_selected.square().mean()
                self.assertEqual(dense_output.logits.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(dense_output.logits).all()))
                dense_loss.backward()
                dense_gradient = actor.residual_output.weight.grad
                assert dense_gradient is not None
                dense_gradient = dense_gradient.detach().float().clone()
                actor.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_name, dtype=amp_dtype):
                    packed_output = actor.forward_packed_batch(
                        batch, normal, packed_indices, packed_mask
                    )
                    packed_loss = packed_output.logits[packed_mask].square().mean()
                self.assertEqual(packed_output.logits.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(packed_output.logits).all()))
                self.assertTrue(torch.allclose(
                    packed_output.logits[packed_mask],
                    dense_output.logits.gather(1, packed_indices)[packed_mask],
                    atol=2.0e-3,
                    rtol=2.0e-3,
                ))
                packed_loss.backward()
                packed_gradient = actor.residual_output.weight.grad
                assert packed_gradient is not None
                self.assertTrue(bool(torch.isfinite(packed_gradient).all()))
                self.assertTrue(torch.allclose(
                    packed_gradient.detach().float(),
                    dense_gradient,
                    atol=3.0e-3,
                    rtol=3.0e-3,
                ))
                probabilities = dense_output.logits.softmax(dim=-1)
                nonforced = batch.legal_mask.sum(dim=-1) > 1
                self.assertTrue(torch.allclose(
                    probabilities[nonforced].gather(
                        1, normal[nonforced, None]
                    ).squeeze(1),
                    torch.full(
                        (int(nonforced.sum()),), 0.9, device=device_name
                    ),
                    atol=2.0e-5,
                    rtol=0.0,
                ))

    def test_central_value_critic_is_training_only_and_parameter_isolated(self) -> None:
        actor = V5PublicActor(tiny_actor_config())
        critic = V5CentralStateValueCritic(
            V5CriticConfig(
                privileged_features=24,
                d_model=32,
                hidden_layers=2,
                player_count_embedding=8,
            )
        )
        assert_actor_critic_parameter_isolation(actor, critic)
        actor_before = {
            name: value.detach().clone() for name, value in actor.named_parameters()
        }
        values = critic(torch.randn(3, 24), torch.tensor([4, 7, 10]))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(torch.equal(values, torch.zeros_like(values)))
        values.sum().backward()
        with torch.no_grad():
            for parameter in critic.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.01)
        self.assertTrue(
            all(
                torch.equal(actor_before[name], value)
                for name, value in actor.named_parameters()
            )
        )
        self.assertEqual(tuple(critic(torch.randn(2, 24)).shape), (2,))

    def test_nonforced_missing_or_illegal_normal_fails_closed(self) -> None:
        legal = torch.zeros(1, V5_ACTION_COUNT, dtype=torch.bool)
        legal[0, [3, 4]] = True
        with self.assertRaisesRegex(ValueError, "normal action must be legal"):
            normal_prior_logits(legal, torch.tensor([-1]))
        with self.assertRaisesRegex(ValueError, "normal action must be legal"):
            normal_prior_logits(legal, torch.tensor([5]))


if __name__ == "__main__":
    unittest.main()
