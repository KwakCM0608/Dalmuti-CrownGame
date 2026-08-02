from __future__ import annotations

"""Evaluation-aligned all-candidate MAPPO collection for DALMUTI V5.

The evaluator fixes the candidate physical identities from the initial seats
for a complete five-act match.  Every one of those identities samples from
the shared public Actor; every remaining identity uses the exact production
Normal policy.  Actor observations and privileged critic states deliberately
cross separate callback and serialization boundaries.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from v4_collect_fixed_match_ppo import (
    ACTS_PER_MATCH,
    DEFAULT_PAIRWISE_COEFFICIENT,
    evaluation_candidate_initial_seats,
    evaluator_group_reward_components,
)
from v4_collect_ppo import (
    NAMESPACE_CHARACTERS,
    _derive_uint32,
    _keyed_uniform,
    masked_categorical_probabilities,
    sample_masked_categorical,
)
from v4_env import (
    ACTION_COUNT,
    PRIVILEGED_STATE_SIZE,
    DalmutiScalarEnv,
    V4ActorObservation,
    V4EnvironmentObservation,
)
from v5_gae import V5GAEResult, compute_smdp_gae


V5_MAPPO_COLLECTION_CONTRACT = "dalmuti-v5-all-candidate-fixed-match-mappo-v1"
V5_MAPPO_REWARD_CONTRACT = "(candidate-mean-chip-normal-mean-chip+0.25*(pairwise-rate-0.5))/5"
V5_MATCH_PROVENANCE_CONTRACT = "global-match-ordinal-bijective-uint32-seed-v2"
V5_BEHAVIOR_TEMPERATURE = 1.0
V5_BEHAVIOR_EPSILON_FLOOR = 0.0
V5_MATCH_STRATUM_COUNT = 7
V5_MAX_MATCH_INDEX_EXCLUSIVE = (
    (0xFFFF_FFFF - (V5_MATCH_STRATUM_COUNT - 1)) // V5_MATCH_STRATUM_COUNT
) + 1

ActorBatch = Callable[
    [Sequence[object], Sequence[int]],
    torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray],
]
CriticBatch = Callable[
    [Sequence[torch.Tensor]], torch.Tensor | np.ndarray | Sequence[float]
]
PublicEncoder = Callable[[V4ActorObservation], object]


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_v5_collection_seed_namespace(
    run_namespace: str,
    seed_base: int,
) -> None:
    if (
        not isinstance(run_namespace, str)
        or not 1 <= len(run_namespace) <= 128
        or run_namespace[0] not in NAMESPACE_CHARACTERS - {".", "_", "-"}
        or any(character not in NAMESPACE_CHARACTERS for character in run_namespace)
        or isinstance(seed_base, bool)
        or not isinstance(seed_base, int)
        or not 0 <= seed_base <= 0xFFFF_FFFF
    ):
        raise ValueError("V5 collection seed namespace is invalid")


def v5_collection_match_ordinal(
    player_count: int,
    match_index: int,
) -> int:
    """Map one p4..p10 match coordinate injectively into one uint32 ordinal."""

    if (
        isinstance(player_count, bool)
        or not isinstance(player_count, int)
        or not 4 <= player_count <= 10
        or isinstance(match_index, bool)
        or not isinstance(match_index, int)
        or not 0 <= match_index < V5_MAX_MATCH_INDEX_EXCLUSIVE
    ):
        raise ValueError("V5 collection match provenance coordinate is invalid")
    return match_index * V5_MATCH_STRATUM_COUNT + (player_count - 4)


@lru_cache(maxsize=256)
def v5_collection_seed_permutation_parameters(
    run_namespace: str,
    seed_base: int,
) -> tuple[int, int]:
    """Return the odd affine multiplier and offset sealed to one run."""

    _validate_v5_collection_seed_namespace(run_namespace, seed_base)
    multiplier = _derive_uint32(
        run_namespace,
        seed_base,
        "v5-fixed-match-environment-bijective-multiplier",
    ) | 1
    offset = _derive_uint32(
        run_namespace,
        seed_base,
        "v5-fixed-match-environment-bijective-offset",
    )
    return multiplier, offset


def derive_v5_collection_match_seed(
    run_namespace: str,
    seed_base: int,
    player_count: int,
    match_index: int,
) -> int:
    """Derive a collision-free uint32 seed from one global match ordinal.

    Multiplication by an odd integer is a permutation modulo 2**32.  Since
    ``v5_collection_match_ordinal`` is injective over the accepted p4..p10
    coordinates, every accepted coordinate receives an exactly unique seed.
    """

    ordinal = v5_collection_match_ordinal(player_count, match_index)
    multiplier, offset = v5_collection_seed_permutation_parameters(
        run_namespace,
        seed_base,
    )
    return (multiplier * ordinal + offset) & 0xFFFF_FFFF


@dataclass(frozen=True)
class V5MAPPOCollectionConfig:
    run_namespace: str
    seed_base: int
    match_counts: tuple[tuple[int, int], ...]
    match_start: int = 0
    match_shard_count: int = 1
    match_shard_index: int = 0
    temperature: float = V5_BEHAVIOR_TEMPERATURE
    epsilon_floor: float = V5_BEHAVIOR_EPSILON_FLOOR
    pairwise_coefficient: float = DEFAULT_PAIRWISE_COEFFICIENT
    gae_lambda: float = 0.95
    require_all_player_counts: bool = False
    lane_count: int = 32

    def __post_init__(self) -> None:
        namespace = self.run_namespace
        if (
            not isinstance(namespace, str)
            or not 1 <= len(namespace) <= 128
            or namespace[0] not in NAMESPACE_CHARACTERS - {".", "_", "-"}
            or any(character not in NAMESPACE_CHARACTERS for character in namespace)
        ):
            raise ValueError("run_namespace must use 1..128 safe ASCII characters")
        if (
            isinstance(self.seed_base, bool)
            or not isinstance(self.seed_base, int)
            or not 0 <= self.seed_base <= 0xFFFF_FFFF
        ):
            raise ValueError("seed_base must be a uint32 integer")
        if not isinstance(self.match_counts, tuple) or not self.match_counts:
            raise ValueError("match_counts must be a non-empty tuple")
        seen: list[int] = []
        for item in self.match_counts:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("match_counts entries must be (player_count, count)")
            player_count, count = item
            if (
                isinstance(player_count, bool)
                or not isinstance(player_count, int)
                or not 4 <= player_count <= 10
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise ValueError("match_counts must bind p4..p10 to positive counts")
            seen.append(player_count)
        if seen != sorted(set(seen)):
            raise ValueError("match_counts player counts must be sorted and unique")
        for name in ("match_start", "match_shard_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.match_shard_count, bool)
            or not isinstance(self.match_shard_count, int)
            or self.match_shard_count < 1
            or self.match_shard_index >= self.match_shard_count
        ):
            raise ValueError("match shard index/count are invalid")
        if any(
            self.match_start + count > V5_MAX_MATCH_INDEX_EXCLUSIVE
            for _, count in self.match_counts
        ):
            raise ValueError(
                "V5 collection match indexes must fit the global uint32 ordinal"
            )
        if (
            isinstance(self.lane_count, bool)
            or not isinstance(self.lane_count, int)
            or self.lane_count < 1
        ):
            raise ValueError("lane_count must be a positive integer")
        if float(self.temperature) != V5_BEHAVIOR_TEMPERATURE:
            raise ValueError("V5 collection requires canonical temperature=1.0")
        if float(self.epsilon_floor) != V5_BEHAVIOR_EPSILON_FLOOR:
            raise ValueError("V5 collection requires canonical epsilon_floor=0.0")
        if float(self.pairwise_coefficient) != DEFAULT_PAIRWISE_COEFFICIENT:
            raise ValueError("V5 collection requires canonical pairwise coefficient 0.25")
        if float(self.gae_lambda) != 0.95:
            raise ValueError("V5 collection requires canonical gae_lambda=0.95")


@dataclass(frozen=True)
class V5DecisionRecord:
    public: object
    privileged_state: np.ndarray
    actor_id: int
    act: int
    normal_action: int
    action: int
    old_log_probability: float
    old_value: float
    entropy: float
    selected_probability: float
    forced: bool
    reward_to_next: float = 0.0
    done: bool = False
    next_decision: int = -1


@dataclass(frozen=True)
class V5ActOutcome:
    act: int
    candidate_mean_chip: float
    normal_mean_chip: float
    chip_difference: float
    pairwise_before: int
    pairwise_comparisons: int
    pairwise_rate: float
    pairwise_centered: float
    team_reward: float


@dataclass(frozen=True)
class V5MatchRecord:
    player_count: int
    match_index: int
    seed: int
    initial_order: tuple[int, ...]
    candidate_initial_seats: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    decisions: tuple[V5DecisionRecord, ...]
    act_outcomes: tuple[V5ActOutcome, ...]


@dataclass(frozen=True)
class V5MAPPOCollection:
    config: V5MAPPOCollectionConfig
    matches: tuple[V5MatchRecord, ...]
    match_offsets: np.ndarray
    candidate_bitsets: np.ndarray
    player_counts: np.ndarray
    decision_actor_ids: np.ndarray
    decision_acts: np.ndarray
    normal_actions: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards_to_next: np.ndarray
    done: np.ndarray
    forced: np.ndarray
    next_decision: np.ndarray
    gae: V5GAEResult

    @property
    def decision_count(self) -> int:
        return int(self.actions.size)

    @property
    def nonforced_decision_count(self) -> int:
        return int(np.logical_not(self.forced).sum())


@dataclass(frozen=True)
class V5PublishedCollection:
    target: Path
    manifest_sha256: str
    matches: int
    decisions: int
    nonforced_decisions: int


def canonicalize_v5_privileged_state(value: torch.Tensor) -> torch.Tensor:
    """Apply the critic's canonical float16 storage boundary losslessly.

    Collection and replay must evaluate the critic on the same quantized state
    that is persisted in the private shard.  The returned tensor is float32 so
    model numerics remain stable; only the critic-only storage precision is
    reduced.
    """

    if not isinstance(value, torch.Tensor) or value.shape != (
        PRIVILEGED_STATE_SIZE,
    ):
        raise TypeError("V5 critic runtime requires 512-vector states")
    state = value.detach().cpu().to(dtype=torch.float32)
    if not bool(torch.isfinite(state).all()):
        raise ValueError("V5 privileged state contains a non-finite value")
    quantized = state.to(dtype=torch.float16).to(dtype=torch.float32)
    if not bool(torch.isfinite(quantized).all()):
        raise ValueError("V5 privileged state escaped float16 storage range")
    return quantized


class V5TorchInferenceRuntime:
    """Batched inference adapter with strict Actor/critic input separation."""

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        *,
        device: str | torch.device,
    ) -> None:
        from v5_model import (
            assert_actor_critic_parameter_isolation,
            configure_v5_policy_numerics,
        )

        assert_actor_critic_parameter_isolation(actor, critic)
        for label, module in (("actor", actor), ("critic", critic)):
            dropout = float(getattr(getattr(module, "config", None), "dropout", 0.0))
            if dropout != 0.0:
                raise ValueError(f"V5 collection requires {label} dropout=0.0")
        self.device = torch.device(device)
        self.policy_numerics = configure_v5_policy_numerics(self.device)
        self.actor = actor.to(self.device).eval()
        self.critic = critic.to(self.device).eval()

    def actor_batch(
        self, observations: Sequence[object], normal_actions: Sequence[int]
    ) -> np.ndarray:
        from v5_public import (
            V5PublicObservation,
            stack_v5_actor_public_features,
            tensorize_v5_public_observation,
        )

        if not observations:
            return np.empty((0, ACTION_COUNT), dtype=np.float64)
        if len(observations) != len(normal_actions) or any(
            type(value) is not V5PublicObservation for value in observations
        ):
            raise TypeError("V5 Actor runtime accepts only matched V5 public observations")
        public_batch = stack_v5_actor_public_features(
            [
                tensorize_v5_public_observation(value)  # type: ignore[arg-type]
                for value in observations
            ],
            device=self.device,
        )
        normal = torch.tensor(
            normal_actions, dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            if hasattr(self.actor, "forward_packed_batch"):
                packed = self.actor.forward_packed_batch(public_batch, normal)
                full = torch.full(
                    (len(observations), ACTION_COUNT),
                    -1.0e9,
                    dtype=packed.logits.dtype,
                    device=self.device,
                )
                rows, positions = packed.action_mask.nonzero(as_tuple=True)
                full[
                    rows, packed.action_indices[rows, positions]
                ] = packed.logits[rows, positions]
                logits = full
            else:
                logits = self.actor(public_batch, normal)
        if logits.shape != (len(observations), ACTION_COUNT):
            raise RuntimeError("V5 Actor returned an invalid collection logit shape")
        return logits.detach().cpu().to(torch.float64).numpy()

    def critic_batch(self, states: Sequence[torch.Tensor]) -> np.ndarray:
        if not states:
            return np.empty((0,), dtype=np.float64)
        batch = torch.stack(
            tuple(canonicalize_v5_privileged_state(value) for value in states)
        ).to(
            device=self.device, dtype=torch.float32
        )
        counts = batch[:, 0].round().to(dtype=torch.long)
        if not bool(((counts >= 4) & (counts <= 10)).all()):
            raise ValueError("privileged state did not encode a valid player count")
        with torch.inference_mode():
            values = self.critic(batch, counts)
        if values.shape != (len(states),):
            raise RuntimeError("V5 critic returned an invalid collection value shape")
        return values.detach().cpu().to(torch.float64).numpy()


@dataclass
class _MutableDecision:
    public: object
    privileged_state: np.ndarray
    actor_id: int
    act: int
    normal_action: int
    action: int
    old_log_probability: float
    old_value: float
    entropy: float
    selected_probability: float
    forced: bool
    reward_to_next: float = 0.0
    done: bool = False
    next_decision: int = -1


@dataclass
class _Lane:
    player_count: int
    match_index: int
    seed: int
    env: DalmutiScalarEnv
    initial_order: tuple[int, ...]
    candidate_initial_seats: tuple[int, ...]
    candidate_ids: frozenset[int]
    observation: V4EnvironmentObservation
    decisions: list[_MutableDecision]
    act_outcomes: list[V5ActOutcome]
    decision_index: int = 0


def evaluator_team_act_reward(
    finish_order: Sequence[int],
    chip_awards: Mapping[int, int] | Mapping[str, int],
    candidate_ids: Sequence[int],
    *,
    pairwise_coefficient: float = DEFAULT_PAIRWISE_COEFFICIENT,
) -> V5ActOutcome:
    """Return the exact evaluator-aligned team reward for one act."""

    (
        candidate_mean,
        normal_mean,
        chip_difference,
        before,
        comparisons,
        rate,
        centered,
    ) = evaluator_group_reward_components(finish_order, chip_awards, candidate_ids)
    team_reward = (
        chip_difference + float(pairwise_coefficient) * centered
    ) / ACTS_PER_MATCH
    if not math.isfinite(team_reward):
        raise RuntimeError("team act reward must be finite")
    return V5ActOutcome(
        act=0,
        candidate_mean_chip=candidate_mean,
        normal_mean_chip=normal_mean,
        chip_difference=chip_difference,
        pairwise_before=before,
        pairwise_comparisons=comparisons,
        pairwise_rate=rate,
        pairwise_centered=centered,
        team_reward=team_reward,
    )


def _as_logits(value: object, batch_size: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        rows = [
            row.detach().cpu().numpy() if isinstance(row, torch.Tensor) else np.asarray(row)
            for row in value  # type: ignore[union-attr]
        ]
        array = np.stack(rows) if rows else np.empty((0, ACTION_COUNT))
    result = np.asarray(array, dtype=np.float64)
    if result.shape != (batch_size, ACTION_COUNT):
        raise ValueError(f"Actor callback must return [{batch_size}, {ACTION_COUNT}] logits")
    return result


def _as_values(value: object, batch_size: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    result = np.asarray(array, dtype=np.float64).reshape(-1)
    if result.shape != (batch_size,) or not np.isfinite(result).all():
        raise ValueError(f"critic callback must return [{batch_size}] finite values")
    return result


def _new_lane(config: V5MAPPOCollectionConfig, player_count: int, match_index: int) -> _Lane:
    seed = derive_v5_collection_match_seed(
        config.run_namespace,
        config.seed_base,
        player_count,
        match_index,
    )
    env = DalmutiScalarEnv(player_count, acts=ACTS_PER_MATCH, seed=seed, device="cpu")
    initial_order = tuple(int(value) for value in env._order)
    candidate_seats = evaluation_candidate_initial_seats(player_count, match_index)
    candidate_ids = frozenset(initial_order[seat] for seat in candidate_seats)
    if not candidate_ids or len(candidate_ids) >= player_count:
        raise RuntimeError("evaluator schedule must produce a non-empty proper candidate group")
    return _Lane(
        player_count,
        match_index,
        seed,
        env,
        initial_order,
        candidate_seats,
        candidate_ids,
        env.observe(),
        [],
        [],
    )


def _freeze_match(lane: _Lane) -> V5MatchRecord:
    if len(lane.act_outcomes) != ACTS_PER_MATCH:
        raise RuntimeError("every collected match must contain exactly five act outcomes")
    if tuple(item.act for item in lane.act_outcomes) != tuple(range(1, ACTS_PER_MATCH + 1)):
        raise RuntimeError("act outcomes must be ordered one through five")
    indexes_by_actor: dict[int, list[int]] = {actor: [] for actor in lane.candidate_ids}
    for index, decision in enumerate(lane.decisions):
        indexes_by_actor[decision.actor_id].append(index)
    for actor_id, indexes in indexes_by_actor.items():
        if not indexes:
            raise RuntimeError(f"candidate {actor_id} had no recorded decision")
        for left, right in zip(indexes, indexes[1:]):
            lane.decisions[left].next_decision = right
        lane.decisions[indexes[-1]].done = True
    frozen = tuple(
        V5DecisionRecord(
            public=item.public,
            privileged_state=item.privileged_state,
            actor_id=item.actor_id,
            act=item.act,
            normal_action=item.normal_action,
            action=item.action,
            old_log_probability=item.old_log_probability,
            old_value=item.old_value,
            entropy=item.entropy,
            selected_probability=item.selected_probability,
            forced=item.forced,
            reward_to_next=item.reward_to_next,
            done=item.done,
            next_decision=item.next_decision,
        )
        for item in lane.decisions
    )
    return V5MatchRecord(
        player_count=lane.player_count,
        match_index=lane.match_index,
        seed=lane.seed,
        initial_order=lane.initial_order,
        candidate_initial_seats=lane.candidate_initial_seats,
        candidate_ids=tuple(sorted(lane.candidate_ids)),
        decisions=frozen,
        act_outcomes=tuple(lane.act_outcomes),
    )


def _flatten_collection(
    config: V5MAPPOCollectionConfig,
    matches: Sequence[V5MatchRecord],
) -> V5MAPPOCollection:
    offsets = [0]
    flat: list[V5DecisionRecord] = []
    global_next: list[int] = []
    for match in matches:
        start = len(flat)
        flat.extend(match.decisions)
        global_next.extend(
            -1 if item.next_decision < 0 else start + item.next_decision
            for item in match.decisions
        )
        offsets.append(len(flat))
    if not flat:
        raise RuntimeError("collection produced no candidate decisions")
    match_offsets = np.asarray(offsets, dtype=np.uint32)
    candidate_bitsets = np.asarray(
        [sum(1 << actor for actor in match.candidate_ids) for match in matches],
        dtype=np.uint16,
    )
    player_counts = np.asarray([match.player_count for match in matches], dtype=np.uint8)
    actor_ids = np.asarray([item.actor_id for item in flat], dtype=np.uint8)
    acts = np.asarray([item.act for item in flat], dtype=np.uint8)
    normal_actions = np.asarray([item.normal_action for item in flat], dtype=np.uint16)
    actions = np.asarray([item.action for item in flat], dtype=np.uint16)
    old_log_probs = np.asarray([item.old_log_probability for item in flat], dtype=np.float32)
    old_values = np.asarray([item.old_value for item in flat], dtype=np.float32)
    rewards = np.asarray([item.reward_to_next for item in flat], dtype=np.float32)
    done = np.asarray([item.done for item in flat], dtype=np.bool_)
    forced = np.asarray([item.forced for item in flat], dtype=np.bool_)
    next_decision = np.asarray(global_next, dtype=np.int32)
    gae = compute_smdp_gae(
        reward_to_next=rewards,
        next_decision=next_decision,
        done=done,
        old_values=old_values,
        match_offsets=match_offsets,
        decision_actor_ids=actor_ids,
        player_counts=player_counts,
        forced=forced,
        candidate_bitsets=candidate_bitsets,
        gamma=1.0,
        gae_lambda=config.gae_lambda,
        require_all_player_counts=config.require_all_player_counts,
    )
    return V5MAPPOCollection(
        config=config,
        matches=tuple(matches),
        match_offsets=match_offsets,
        candidate_bitsets=candidate_bitsets,
        player_counts=player_counts,
        decision_actor_ids=actor_ids,
        decision_acts=acts,
        normal_actions=normal_actions,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        rewards_to_next=rewards,
        done=done,
        forced=forced,
        next_decision=next_decision,
        gae=gae,
    )


def collect_v5_mappo(
    actor_batch: ActorBatch,
    critic_batch: CriticBatch,
    config: V5MAPPOCollectionConfig,
    *,
    public_encoder: PublicEncoder | None = None,
) -> V5MAPPOCollection:
    """Collect complete all-candidate matches using exact Normal opponents.

    The Actor callback receives only validated V5 public observations plus the
    exact Normal action used by the residual policy.  The critic callback is
    the sole recipient of privileged 512-vectors.
    """

    if public_encoder is None:
        from v5_public import v5_public_from_v4_actor_observation

        public_encoder = v5_public_from_v4_actor_observation

    specs = [
        (player_count, match_index)
        for player_count, count in config.match_counts
        for match_index in range(config.match_start, config.match_start + count)
        if match_index % config.match_shard_count == config.match_shard_index
    ]
    if not specs:
        raise ValueError("requested V5 match shard is empty")
    completed: list[V5MatchRecord] = []
    lanes = [
        _new_lane(config, player_count, match_index)
        for player_count, match_index in specs[: config.lane_count]
    ]
    next_spec = len(lanes)
    while lanes:
        publics = [lane.observation.public for lane in lanes]
        normals = [lane.env.normal_action() for lane in lanes]
        candidate_indexes = [
            index
            for index, lane in enumerate(lanes)
            if lane.env.current_player_id in lane.candidate_ids
        ]
        candidate_publics = tuple(
            public_encoder(publics[index]) for index in candidate_indexes
        )
        candidate_normals = tuple(normals[index] for index in candidate_indexes)
        privileged_states = tuple(
            canonicalize_v5_privileged_state(
                lanes[index].observation.privileged_state
            )
            for index in candidate_indexes
        )
        if candidate_indexes:
            logits = _as_logits(
                actor_batch(candidate_publics, candidate_normals),
                len(candidate_indexes),
            )
            values = _as_values(
                critic_batch(privileged_states), len(candidate_indexes)
            )
        else:
            logits = np.empty((0, ACTION_COUNT), dtype=np.float64)
            values = np.empty((0,), dtype=np.float64)
        candidate_rows = {
            lane_index: (row_logits, float(value), privileged, encoded_public)
            for lane_index, row_logits, value, privileged, encoded_public in zip(
                candidate_indexes,
                logits,
                values,
                privileged_states,
                candidate_publics,
                strict=True,
            )
        }

        replacements: list[_Lane] = []
        for lane_index, (lane, public, normal_action) in enumerate(
            zip(lanes, publics, normals, strict=True)
        ):
            env = lane.env
            actor_id = env.current_player_id
            act = int(env._act)
            if not bool(public.legal_mask[normal_action]):
                raise RuntimeError("exact Normal selected an illegal action")
            if actor_id in lane.candidate_ids:
                row_logits, value, privileged, encoded_public = candidate_rows[lane_index]
                probabilities = masked_categorical_probabilities(
                    row_logits,
                    public.legal_mask,
                    temperature=config.temperature,
                    epsilon_floor=config.epsilon_floor,
                )
                uniform = _keyed_uniform(
                    config.run_namespace,
                    config.seed_base,
                    "v5-all-candidate-action",
                    lane.player_count,
                    lane.match_index,
                    lane.seed,
                    act,
                    actor_id,
                    lane.decision_index,
                )
                action, log_probability, entropy = sample_masked_categorical(
                    probabilities, uniform
                )
                forced = int(public.legal_mask.sum().item()) == 1
                lane.decisions.append(_MutableDecision(
                    public=encoded_public,
                    privileged_state=np.asarray(privileged.numpy(), dtype=np.float16),
                    actor_id=actor_id,
                    act=act,
                    normal_action=normal_action,
                    action=action,
                    old_log_probability=log_probability,
                    old_value=value,
                    entropy=entropy,
                    selected_probability=float(probabilities[action]),
                    forced=forced,
                ))
                lane.decision_index += 1
                behavior_action = action
            else:
                behavior_action = normal_action

            result = env.step(int(behavior_action))
            lane.observation = result.observation
            if result.act_ended:
                act_result = result.info.get("act_result")
                if not isinstance(act_result, Mapping):
                    raise RuntimeError("ended act omitted the exact act result")
                outcome = evaluator_team_act_reward(
                    act_result["finish_order"],  # type: ignore[arg-type]
                    act_result["chip_awards"],  # type: ignore[arg-type]
                    lane.candidate_ids,
                    pairwise_coefficient=config.pairwise_coefficient,
                )
                outcome = V5ActOutcome(act=act, **{
                    name: getattr(outcome, name)
                    for name in (
                        "candidate_mean_chip",
                        "normal_mean_chip",
                        "chip_difference",
                        "pairwise_before",
                        "pairwise_comparisons",
                        "pairwise_rate",
                        "pairwise_centered",
                        "team_reward",
                    )
                })
                lane.act_outcomes.append(outcome)
                # Each candidate identity receives the shared team reward on
                # its last decision spanning this act boundary.
                for candidate_id in lane.candidate_ids:
                    indexes = [
                        index
                        for index, item in enumerate(lane.decisions)
                        if item.actor_id == candidate_id and item.act == act
                    ]
                    if not indexes:
                        raise RuntimeError(
                            f"candidate {candidate_id} had no decision in act {act}"
                        )
                    lane.decisions[indexes[-1]].reward_to_next += outcome.team_reward
            if result.terminated:
                completed.append(_freeze_match(lane))
                if next_spec < len(specs):
                    replacements.append(_new_lane(config, *specs[next_spec]))
                    next_spec += 1
            else:
                replacements.append(lane)
        lanes = replacements
    completed.sort(key=lambda item: (item.player_count, item.match_index))
    return _flatten_collection(config, completed)


def v5_collection_array_partitions(
    collection: V5MAPPOCollection,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the public Actor and private critic array partitions."""

    from v5_public import pack_v5_public_observations

    decisions = [
        decision
        for match in collection.matches
        for decision in match.decisions
    ]
    if len(decisions) != collection.decision_count:
        raise RuntimeError("collection decision objects and structural arrays diverged")
    public_arrays, history_events, history_end = pack_v5_public_observations(
        [item.public for item in decisions]  # type: ignore[list-item]
    )
    actor_arrays = dict(public_arrays)
    actor_arrays.update({
        "match_offsets": collection.match_offsets,
        "candidate_bitsets": collection.candidate_bitsets,
        "player_counts": collection.player_counts,
        "decision_actor_ids": collection.decision_actor_ids,
        "decision_acts": collection.decision_acts,
        "normal_actions": collection.normal_actions,
        "actions": collection.actions,
        "old_log_probs": collection.old_log_probs,
        "old_values": collection.old_values,
        "reward_to_next": collection.rewards_to_next,
        "done": collection.done,
        "forced": collection.forced,
        "next_decision": collection.next_decision,
        "history_events": history_events,
        "history_end": history_end,
        "selected_action_probabilities": np.asarray(
            [item.selected_probability for item in decisions], dtype=np.float32
        ),
        "policy_entropies": np.asarray(
            [item.entropy for item in decisions], dtype=np.float32
        ),
        "advantages": collection.gae.advantages,
        "returns": collection.gae.returns,
        "deltas": collection.gae.deltas,
        "policy_mask": collection.gae.policy_mask,
        "value_mask": collection.gae.value_mask,
        "policy_loss_weights": collection.gae.policy_loss_weights,
        "value_loss_weights": collection.gae.value_loss_weights,
    })
    match_indices = np.asarray(
        [match.match_index for match in collection.matches], dtype=np.uint32
    )
    match_seeds = np.asarray(
        [match.seed for match in collection.matches], dtype=np.uint32
    )
    if (
        match_indices.shape != (len(collection.matches),)
        or match_seeds.shape != (len(collection.matches),)
        or len({
            (int(player), int(index))
            for player, index in zip(
                collection.player_counts, match_indices, strict=True
            )
        }) != len(collection.matches)
        or np.unique(match_seeds).size != len(collection.matches)
    ):
        raise RuntimeError("collection match provenance is not globally unique")
    privileged = {
        "match_indices": match_indices,
        "match_seeds": match_seeds,
        "privileged_states": np.stack(
            [item.privileged_state for item in decisions]
        ).astype(np.float16, copy=False)
    }
    if privileged["privileged_states"].shape != (
        collection.decision_count,
        PRIVILEGED_STATE_SIZE,
    ):
        raise RuntimeError("privileged critic partition has an invalid shape")
    return actor_arrays, privileged


def publish_v5_mappo_collection(
    target: str | Path,
    collection: V5MAPPOCollection,
    *,
    behavior_actor_sha256: str,
    behavior_actor_manifest_sha256: str,
    behavior_critic_sha256: str,
    metadata: Mapping[str, object] | None = None,
) -> V5PublishedCollection:
    """Exclusively publish one immutable mmap-native V5 training shard."""

    from v5_contract import V5_PUBLIC_CONTRACT_SHA256
    from v5_dataset import publish_v5_shard

    actor_arrays, privileged_arrays = v5_collection_array_partitions(collection)
    config = collection.config
    actor_sha = _require_sha256(behavior_actor_sha256, "behavior_actor_sha256")
    actor_manifest_sha = _require_sha256(
        behavior_actor_manifest_sha256, "behavior_actor_manifest_sha256"
    )
    critic_sha = _require_sha256(
        behavior_critic_sha256, "behavior_critic_sha256"
    )
    base_metadata: dict[str, object] = {
        "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
        "rewardContract": V5_MAPPO_REWARD_CONTRACT,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "runNamespace": config.run_namespace,
        "seedBase": config.seed_base,
        "matchCounts": {str(player): count for player, count in config.match_counts},
        "matchStart": config.match_start,
        "matchShardCount": config.match_shard_count,
        "matchShardIndex": config.match_shard_index,
        "matchProvenanceContract": V5_MATCH_PROVENANCE_CONTRACT,
        "actsPerMatch": ACTS_PER_MATCH,
        "behaviorTemperature": config.temperature,
        "behaviorEpsilonFloor": config.epsilon_floor,
        "pairwiseCoefficient": config.pairwise_coefficient,
        "gamma": 1.0,
        "gaeLambda": config.gae_lambda,
        "allCandidateDecisionsRecorded": True,
        "normalOpponentsExact": True,
        "candidatePhysicalIdsFixedForCompleteMatch": True,
        "behaviorActorSha256": actor_sha,
        "behaviorActorManifestSha256": actor_manifest_sha,
        "behaviorCriticSha256": critic_sha,
    }
    if metadata is not None:
        overlap = set(base_metadata) & set(metadata)
        if overlap:
            raise ValueError(
                f"caller metadata cannot replace collection contract field: {sorted(overlap)[0]}"
            )
        base_metadata.update(metadata)
    target_path = Path(target).resolve()
    digest = publish_v5_shard(
        target_path,
        actor_arrays,
        privileged_arrays,
        metadata=base_metadata,
        action_count=ACTION_COUNT,
    )
    return V5PublishedCollection(
        target=target_path,
        manifest_sha256=digest,
        matches=len(collection.matches),
        decisions=collection.decision_count,
        nonforced_decisions=collection.nonforced_decision_count,
    )


__all__ = [
    "ActorBatch",
    "CriticBatch",
    "PublicEncoder",
    "V5ActOutcome",
    "V5DecisionRecord",
    "V5MAPPOCollection",
    "V5MAPPOCollectionConfig",
    "V5MatchRecord",
    "V5PublishedCollection",
    "V5TorchInferenceRuntime",
    "V5_BEHAVIOR_EPSILON_FLOOR",
    "V5_BEHAVIOR_TEMPERATURE",
    "V5_MAPPO_COLLECTION_CONTRACT",
    "V5_MAPPO_REWARD_CONTRACT",
    "V5_MATCH_STRATUM_COUNT",
    "V5_MAX_MATCH_INDEX_EXCLUSIVE",
    "V5_MATCH_PROVENANCE_CONTRACT",
    "collect_v5_mappo",
    "derive_v5_collection_match_seed",
    "evaluator_team_act_reward",
    "publish_v5_mappo_collection",
    "v5_collection_array_partitions",
    "v5_collection_match_ordinal",
    "v5_collection_seed_permutation_parameters",
]
