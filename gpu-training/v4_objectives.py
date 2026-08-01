from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from v4_model import V4_ACTION_COUNT, V4_MASKED_LOGIT


def _validate_action_matrix(
    values: torch.Tensor, legal_masks: torch.Tensor, label: str
) -> None:
    if values.shape != legal_masks.shape or values.shape[-1] != V4_ACTION_COUNT:
        raise ValueError(f"{label} and legal masks must end in 236 actions")
    if legal_masks.dtype != torch.bool:
        raise ValueError("legal masks must use torch.bool")
    if not legal_masks.any(dim=-1).all():
        raise ValueError("every policy row requires a legal action")


def masked_log_probabilities(
    logits: torch.Tensor, legal_masks: torch.Tensor
) -> torch.Tensor:
    _validate_action_matrix(logits, legal_masks, "logits")
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must use a floating-point dtype")
    masked_value = max(V4_MASKED_LOGIT, torch.finfo(logits.dtype).min / 2.0)
    return F.log_softmax(logits.masked_fill(~legal_masks, masked_value), dim=-1)


def masked_probabilities(
    logits: torch.Tensor, legal_masks: torch.Tensor
) -> torch.Tensor:
    probabilities = masked_log_probabilities(logits, legal_masks).exp()
    return probabilities.masked_fill(~legal_masks, 0.0)


def expected_action_q(
    q_values: torch.Tensor,
    policy_logits: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    _validate_action_matrix(q_values, legal_masks, "Q values")
    _validate_action_matrix(policy_logits, legal_masks, "policy logits")
    probabilities = masked_probabilities(policy_logits, legal_masks)
    safe_q_values = q_values.masked_fill(~legal_masks, 0.0)
    return (probabilities * safe_q_values).sum(dim=-1)


def expected_sarsa_lambda_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    q_values: torch.Tensor,
    policy_logits: torch.Tensor,
    legal_masks: torch.Tensor,
    *,
    gamma: float = 1.0,
    lambda_: float = 0.95,
    valid_masks: torch.Tensor | None = None,
    bootstrap_expected_q: torch.Tensor | None = None,
    detach: bool = True,
) -> torch.Tensor:
    """Build backward-view Expected-SARSA(lambda) action-Q targets.

    Inputs use time-major `[time, batch, ...]` layout. Q/policy rows describe
    the current states; the expectation at `t + 1` supplies the one-step
    bootstrap for transition `t`. Padded trajectory suffixes are excluded by
    `valid_masks`. A truncated final transition can supply one bootstrap value
    per trajectory.
    """

    if rewards.ndim != 2:
        raise ValueError("rewards must be [time, batch]")
    if dones.dtype != torch.bool or dones.shape != rewards.shape:
        raise ValueError("dones must be bool [time, batch]")
    expected_shape = (*rewards.shape, V4_ACTION_COUNT)
    if q_values.shape != expected_shape:
        raise ValueError("Q values must be [time, batch, 236]")
    if policy_logits.shape != expected_shape or legal_masks.shape != expected_shape:
        raise ValueError("policy logits and legal masks must match Q values")
    gamma_value = float(gamma)
    lambda_value = float(lambda_)
    if not math.isfinite(gamma_value) or gamma_value < 0.0 or gamma_value > 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not math.isfinite(lambda_value) or lambda_value < 0.0 or lambda_value > 1.0:
        raise ValueError("lambda_ must be finite and in [0, 1]")
    if valid_masks is None:
        valid_masks = torch.ones_like(dones)
    if valid_masks.dtype != torch.bool or valid_masks.shape != rewards.shape:
        raise ValueError("valid masks must be bool [time, batch]")
    # Valid samples must form a prefix so a later state can never follow padding.
    if rewards.shape[0] > 1 and (valid_masks[1:] & ~valid_masks[:-1]).any():
        raise ValueError("trajectory valid masks must be contiguous prefixes")
    if bootstrap_expected_q is None:
        bootstrap_expected_q = rewards.new_zeros(rewards.shape[1])
    if bootstrap_expected_q.shape != (rewards.shape[1],):
        raise ValueError("bootstrap expected Q must have shape [batch]")

    expected = expected_action_q(q_values, policy_logits, legal_masks)
    if detach:
        expected = expected.detach()
        rewards = rewards.detach()
        bootstrap_expected_q = bootstrap_expected_q.detach()
    targets = torch.zeros_like(rewards)
    running = bootstrap_expected_q
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        if time_index + 1 < rewards.shape[0]:
            next_is_valid = valid_masks[time_index + 1]
            one_step_bootstrap = torch.where(
                next_is_valid,
                expected[time_index + 1],
                bootstrap_expected_q,
            )
        else:
            one_step_bootstrap = bootstrap_expected_q
        discount = gamma_value * (~dones[time_index]).to(rewards.dtype)
        candidate = rewards[time_index] + discount * (
            (1.0 - lambda_value) * one_step_bootstrap
            + lambda_value * running
        )
        current_valid = valid_masks[time_index]
        targets[time_index] = torch.where(
            current_valid, candidate, torch.zeros_like(candidate)
        )
        running = torch.where(current_valid, candidate, running)
    return targets


@dataclass(frozen=True)
class V4PolicyLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    mean_q_boost: torch.Tensor


def vrpo_clipped_policy_loss(
    policy_logits: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    q_values: torch.Tensor | None = None,
    behavior_policy_logits: torch.Tensor | None = None,
    q_boost_coefficient: float = 1.0,
    clip_ratio: float = 0.15,
    entropy_coefficient: float = 0.01,
    normalize_advantages: bool = False,
) -> V4PolicyLoss:
    """Clipped PPO with a detached centered action-Q advantage (VRPO utility)."""

    if policy_logits.ndim != 2 or policy_logits.shape[-1] != V4_ACTION_COUNT:
        raise ValueError("policy logits must be [batch, 236]")
    batch_size = policy_logits.shape[0]
    _validate_action_matrix(policy_logits, legal_masks, "policy logits")
    if actions.dtype != torch.long or actions.shape != (batch_size,):
        raise ValueError("actions must be torch.long [batch]")
    for value, label in (
        (old_action_log_probs, "old action log probabilities"),
        (advantages, "advantages"),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{label} must have shape [batch]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} must be finite")
    chosen_is_legal = legal_masks.gather(1, actions[:, None]).squeeze(1)
    if not chosen_is_legal.all():
        raise ValueError("sampled actions must be legal")
    clip_value = float(clip_ratio)
    entropy_value = float(entropy_coefficient)
    q_coefficient = float(q_boost_coefficient)
    if not math.isfinite(clip_value) or clip_value <= 0.0 or clip_value >= 1.0:
        raise ValueError("clip_ratio must be finite and in (0, 1)")
    if not math.isfinite(entropy_value) or entropy_value < 0.0:
        raise ValueError("entropy coefficient must be finite and non-negative")
    if not math.isfinite(q_coefficient) or q_coefficient < 0.0:
        raise ValueError("Q boost coefficient must be finite and non-negative")

    log_probabilities = masked_log_probabilities(policy_logits, legal_masks)
    action_log_probs = log_probabilities.gather(1, actions[:, None]).squeeze(1)
    combined_advantages = advantages
    q_boost = torch.zeros_like(advantages)
    if q_values is not None:
        if q_values.shape != policy_logits.shape or not torch.isfinite(
            q_values.masked_fill(~legal_masks, 0.0)
        ).all():
            raise ValueError("Q values must be finite [batch, 236]")
        baseline_logits = (
            behavior_policy_logits
            if behavior_policy_logits is not None
            else policy_logits.detach()
        )
        if baseline_logits.shape != policy_logits.shape:
            raise ValueError("behavior policy logits must match policy logits")
        detached_q = q_values.detach()
        q_baseline = expected_action_q(
            detached_q, baseline_logits.detach(), legal_masks
        )
        q_taken = detached_q.gather(1, actions[:, None]).squeeze(1)
        q_boost = q_taken - q_baseline
        combined_advantages = combined_advantages + q_coefficient * q_boost
    if normalize_advantages and batch_size > 1:
        combined_advantages = (
            combined_advantages - combined_advantages.mean()
        ) / combined_advantages.std(unbiased=False).clamp_min(1.0e-6)
    combined_advantages = combined_advantages.detach()

    log_ratio = action_log_probs - old_action_log_probs
    ratio = log_ratio.exp()
    unclipped = ratio * combined_advantages
    clipped = ratio.clamp(1.0 - clip_value, 1.0 + clip_value) * combined_advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    probabilities = log_probabilities.exp().masked_fill(~legal_masks, 0.0)
    entropy = -(probabilities * log_probabilities.masked_fill(~legal_masks, 0.0)).sum(
        dim=-1
    ).mean()
    total_loss = policy_loss - entropy_value * entropy
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_value).to(policy_logits.dtype).mean()
    return V4PolicyLoss(
        loss=total_loss,
        policy_loss=policy_loss,
        entropy=entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        mean_q_boost=q_boost.mean(),
    )


def masked_behavior_cloning_loss(
    policy_logits: torch.Tensor,
    legal_masks: torch.Tensor,
    expert_actions: torch.Tensor,
) -> torch.Tensor:
    _validate_action_matrix(policy_logits, legal_masks, "policy logits")
    if expert_actions.dtype != torch.long or expert_actions.shape != (
        policy_logits.shape[0],
    ):
        raise ValueError("expert actions must be torch.long [batch]")
    if not legal_masks.gather(1, expert_actions[:, None]).all():
        raise ValueError("expert actions must be legal")
    return F.nll_loss(
        masked_log_probabilities(policy_logits, legal_masks), expert_actions
    )


def action_q_regression_loss(
    q_values: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    *,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    _validate_action_matrix(q_values, legal_masks, "Q values")
    batch_size = q_values.shape[0]
    if actions.dtype != torch.long or actions.shape != (batch_size,):
        raise ValueError("actions must be torch.long [batch]")
    if targets.shape != (batch_size,) or not torch.isfinite(targets).all():
        raise ValueError("Q targets must be finite [batch]")
    if not legal_masks.gather(1, actions[:, None]).all():
        raise ValueError("critic actions must be legal")
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber delta must be positive and finite")
    predictions = q_values.gather(1, actions[:, None]).squeeze(1)
    return F.huber_loss(predictions, targets.detach(), delta=delta)


__all__ = [
    "V4PolicyLoss",
    "action_q_regression_loss",
    "expected_action_q",
    "expected_sarsa_lambda_targets",
    "masked_behavior_cloning_loss",
    "masked_log_probabilities",
    "masked_probabilities",
    "vrpo_clipped_policy_loss",
]
