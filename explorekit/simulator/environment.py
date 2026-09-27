"""Синтетическая среда ExploreKit и контракт ``make_replication``.

Минимальный контракт Полины (не менять без согласования)::

    make_replication(seed) -> Replication(ope_input, true_value)

Дополнительно (подхватывает участник 4)::

    logs, exploration_share, cold_item_speedup, ctr_cost

``chosen_item`` везде int-индекс в ``[0, n_actions)``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from explorekit.cold_start import ColdItemBooster
from explorekit.logging import DecisionLog
from explorekit.ope.base import OPEInput
from explorekit.policies import BasePolicy, Decision, EpsilonGreedyPolicy
from explorekit.simulator.base_ranker import BaseRanker
from explorekit.simulator.items import ItemCatalog
from explorekit.simulator.metrics import cold_item_speedup, reached_n_share, time_to_n
from explorekit.simulator.reward import TrueReward, calibrate_intercept, sigmoid
from explorekit.simulator.users import UserGenerator

# Adapter участника 4 читает эти константы, чтобы выровнять policy.n_actions
# и policy.context_dim с симулятором.
N_ACTIONS = 80
N_CONTEXT_FEATURES = 8
N_DAYS = 20
TARGET_CTR = 0.05
LOGGING_EPSILON = 0.5
LOGGING_POLICY_NAME = "epsilon_greedy_logging"

__all__ = [
    "N_ACTIONS",
    "N_CONTEXT_FEATURES",
    "N_DAYS",
    "SimulatorReplication",
    "World",
    "build_world",
    "make_replication",
]


@dataclass
class World:
    """Неизменяемый после сборки мир: каталог, пользователи, истина, ранкер."""

    catalog: ItemCatalog
    users: UserGenerator
    reward: TrueReward
    ranker: BaseRanker
    n_days: int


@dataclass
class RoundBatch:
    """Предсэмплированные раунды. Политики бегают по одной и той же ленте."""

    context: np.ndarray
    segment_ids: np.ndarray
    segment_names: list[str]
    day: np.ndarray
    true_logits: np.ndarray
    true_p: np.ndarray
    base_scores: np.ndarray
    candidate_items: list[np.ndarray]


@dataclass
class SimulatorReplication:
    """Расширенный Replication: обязательные поля Полины + метрики участника 3."""

    ope_input: OPEInput
    true_value: float
    logs: list[DecisionLog] = field(default_factory=list)
    exploration_share: float | None = None
    cold_item_speedup: float | None = None
    ctr_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_world(
    rng: np.random.Generator,
    *,
    n_actions: int = N_ACTIONS,
    feature_dim: int = N_CONTEXT_FEATURES,
    n_days: int = N_DAYS,
    base_score_noise: float = 0.3,
    cold_penalty: float = 1.2,
    target_ctr: float = TARGET_CTR,
    seed: int = 0,
) -> World:
    """Собирает каталог, пользователей и скрытую reward-функцию."""

    catalog = ItemCatalog(n_actions=n_actions, feature_dim=feature_dim, rng=rng)
    users = UserGenerator(feature_dim=feature_dim, rng=rng)
    affinity = rng.normal(scale=0.35, size=(users.n_segments, catalog.n_categories))
    intercept = calibrate_intercept(
        catalog,
        users,
        affinity,
        np.random.default_rng(seed + 7),
        target_ctr=target_ctr,
    )
    reward = TrueReward(
        item_features=catalog.features,
        category_ids=catalog.category_ids,
        affinity=affinity,
        intercept=intercept,
    )
    ranker = BaseRanker(noise_scale=base_score_noise, cold_penalty=cold_penalty)
    return World(
        catalog=catalog, users=users, reward=reward, ranker=ranker, n_days=n_days
    )


def _days_for_rounds(n_rounds: int, n_days: int) -> np.ndarray:
    if n_rounds <= 0 or n_days <= 0:
        raise ValueError("n_rounds and n_days must be positive")
    return np.minimum((np.arange(n_rounds) * n_days) // n_rounds, n_days - 1)


def generate_batch(
    world: World,
    n_rounds: int,
    rng: np.random.Generator,
    *,
    max_age_days: int,
) -> RoundBatch:
    """Сэмплирует контексты, true P(click) и base scores на весь прогон."""

    day = _days_for_rounds(n_rounds, world.n_days)
    context, segment_ids, names = world.users.sample_batch(n_rounds, rng)
    true_logits = world.reward.logits(context, segment_ids)
    true_p = sigmoid(true_logits)

    ages = np.stack([world.catalog.age_days(int(d)) for d in day], axis=0)
    base_scores = world.ranker.scores(
        true_logits, age_days=ages, max_age_days=max_age_days, rng=rng
    )

    candidate_items: list[np.ndarray] = []
    unavailable = -1e9
    for t in range(n_rounds):
        cand = world.catalog.available_at(int(day[t]))
        candidate_items.append(cand)
        mask = np.ones(world.catalog.n_actions, dtype=bool)
        mask[cand] = False
        base_scores[t, mask] = unavailable

    return RoundBatch(
        context=context,
        segment_ids=segment_ids,
        segment_names=names,
        day=day,
        true_logits=true_logits,
        true_p=true_p,
        base_scores=base_scores,
        candidate_items=candidate_items,
    )


def _epsilon_greedy_dist(
    scores: np.ndarray, epsilon: float, candidates: np.ndarray
) -> np.ndarray:
    dist = np.zeros(scores.shape[0], dtype=float)
    if candidates.size == 0:
        raise ValueError("candidate set is empty")
    dist[candidates] = float(epsilon) / float(candidates.size)
    greedy = int(candidates[np.argmax(scores[candidates])])
    dist[greedy] += 1.0 - float(epsilon)
    return dist


def _policy_distribution_matrix(
    policy: BasePolicy,
    batch: RoundBatch,
) -> np.ndarray:
    """``(n_rounds, n_actions)`` вероятностей evaluation policy.

    Кандидат-наборы меняются по дням, поэтому батчим только одинаковые наборы.
    """

    n_rounds, n_actions = batch.base_scores.shape
    dist = np.zeros((n_rounds, n_actions), dtype=float)
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, cand in enumerate(batch.candidate_items):
        groups.setdefault(tuple(int(x) for x in cand.tolist()), []).append(i)

    eval_policy = copy.deepcopy(policy)
    if hasattr(eval_policy, "exploration_subset"):
        eval_policy.exploration_subset = None

    for key, rows in groups.items():
        idx = np.asarray(rows, dtype=int)
        cand = np.asarray(key, dtype=int)
        part = eval_policy.action_distribution(
            batch.context[idx],
            batch.base_scores[idx],
            candidate_items=cand,
        )
        dist[idx] = np.asarray(part, dtype=float)
    return dist


def _expected_ctr(action_dist: np.ndarray, true_p: np.ndarray) -> float:
    return float(np.sum(action_dist * true_p, axis=1).mean())


def _greedy_policy(n_actions: int, seed: int) -> EpsilonGreedyPolicy:
    return EpsilonGreedyPolicy(
        n_actions=n_actions,
        epsilon=0.0,
        min_epsilon=0.0,
        max_epsilon=1.0,
        seed=seed,
    )


def _serving_booster(
    policy: BasePolicy, booster: ColdItemBooster
) -> ColdItemBooster | None:
    """Baseline ε=0 должен остаться чистым exploitation без boost."""

    if hasattr(policy, "epsilon") and float(policy.epsilon) <= 1e-12:
        return None
    return booster


def _evaluation_policy(
    seed: int,
    n_actions: int,
    epsilon: float,
    policy: BasePolicy | None,
    evaluation_policy: BasePolicy | None,
) -> BasePolicy:
    chosen = evaluation_policy if evaluation_policy is not None else policy
    if chosen is not None:
        return chosen
    return EpsilonGreedyPolicy(
        n_actions=n_actions,
        epsilon=epsilon,
        min_epsilon=0.0,
        max_epsilon=1.0,
        seed=seed,
    )


def run_online(
    world: World,
    batch: RoundBatch,
    policy: BasePolicy,
    booster: ColdItemBooster | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Онлайн-прогон политики по уже сэмплированной ленте контекстов."""

    n_rounds, n_actions = batch.base_scores.shape
    impressions = np.zeros(n_actions, dtype=int)
    impressions[: world.catalog.n_warm] = (
        booster.min_impressions if booster is not None else 0
    )
    chosen = np.empty(n_rounds, dtype=int)
    rewards = np.empty(n_rounds, dtype=int)
    is_exploration = np.zeros(n_rounds, dtype=bool)
    is_cold = np.zeros(n_rounds, dtype=bool)
    expected_click = np.empty(n_rounds, dtype=float)

    actor = copy.deepcopy(policy)

    for t in range(n_rounds):
        cand = batch.candidate_items[t]
        available = np.zeros(n_actions, dtype=bool)
        available[cand] = True
        ages = world.catalog.age_days(int(batch.day[t]))
        scores = batch.base_scores[t]
        cold_indices = np.array([], dtype=int)
        cold_mask = np.zeros(n_actions, dtype=bool)
        if booster is not None:
            decision_boost = booster.apply(scores, ages, impressions, available)
            scores = decision_boost.boosted_scores
            cold_indices = decision_boost.cold_indices
            cold_mask = decision_boost.cold_mask
            if hasattr(actor, "exploration_subset"):
                actor.exploration_subset = cold_indices if cold_indices.size else None

        context = batch.context[t]
        decision = actor.select_action(context, scores, cand)
        item = int(decision.chosen_item)
        p_true = float(batch.true_p[t, item])
        reward = int(rng.binomial(1, p_true))
        actor.update(decision, float(reward), context)

        impressions[item] += 1
        chosen[t] = item
        rewards[t] = reward
        is_exploration[t] = bool(decision.is_exploration)
        is_cold[t] = bool(cold_mask[item])
        expected_click[t] = p_true

    return {
        "chosen": chosen,
        "rewards": rewards,
        "is_exploration": is_exploration,
        "is_cold": is_cold,
        "expected_click": expected_click,
        "impressions": impressions,
        "online_ctr": float(rewards.mean()),
        "exploration_share": float(is_exploration.mean()),
    }


def _logging_pass(
    world: World,
    batch: RoundBatch,
    booster: ColdItemBooster,
    rng: np.random.Generator,
    *,
    logging_epsilon: float,
    seed: int,
) -> tuple[list[DecisionLog], np.ndarray, np.ndarray, np.ndarray]:
    """Исторические логи: ε-greedy с большим ε, чтобы все айтемы имели шанс."""

    n_rounds, n_actions = batch.base_scores.shape
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logs: list[DecisionLog] = []
    chosen = np.empty(n_rounds, dtype=int)
    behavior = np.empty(n_rounds, dtype=float)
    rewards = np.empty(n_rounds, dtype=float)
    impressions = np.zeros(n_actions, dtype=int)
    impressions[: world.catalog.n_warm] = booster.min_impressions

    for t in range(n_rounds):
        cand = batch.candidate_items[t]
        dist = _epsilon_greedy_dist(batch.base_scores[t], logging_epsilon, cand)
        item = int(rng.choice(n_actions, p=dist))
        p_true = float(batch.true_p[t, item])
        reward = int(rng.binomial(1, p_true))
        greedy = int(cand[np.argmax(batch.base_scores[t, cand])])
        decision = Decision(
            chosen_item=item,
            propensity=float(dist[item]),
            probabilities=dist,
            is_exploration=item != greedy,
            policy_name=LOGGING_POLICY_NAME,
        )
        day = int(batch.day[t])
        ages = world.catalog.age_days(day)
        available = np.zeros(n_actions, dtype=bool)
        available[cand] = True
        cold = booster.is_cold(ages, impressions, available)
        logs.append(
            DecisionLog.from_decision(
                decision=decision,
                request_id=f"sim-{seed}-{t:06d}",
                user_segment=batch.segment_names[t],
                context=batch.context[t],
                candidate_items=cand,
                base_scores=batch.base_scores[t],
                policy_params={"epsilon": logging_epsilon},
                reward=reward,
                is_cold_item=bool(cold[item]),
                timestamp=start + timedelta(days=day, seconds=t % 86400),
            )
        )
        impressions[item] += 1
        chosen[t] = item
        behavior[t] = float(dist[item])
        rewards[t] = float(reward)

    return logs, chosen, behavior, rewards


def make_replication(
    seed: int,
    n_rounds: int = 5000,
    epsilon: float = 0.1,
    base_score_noise: float = 0.3,
    reward_model_noise: float = 0.25,
    policy: BasePolicy | None = None,
    evaluation_policy: BasePolicy | None = None,
    max_age_days: int = 7,
    min_impressions: int = 20,
    boost_multiplier: float = 2.0,
    time_to_n_threshold: int = 20,
    n_days: int = N_DAYS,
    logging_epsilon: float = LOGGING_EPSILON,
    target_ctr: float = TARGET_CTR,
    cold_penalty: float = 1.2,
    compute_cold_metrics: bool = True,
) -> SimulatorReplication:
    """Один прогон симулятора в контракте Полины + метрики cold-start.

    Args:
        seed: общий seed (``np.random.default_rng``).
        n_rounds: число показов.
        epsilon: ε оцениваемой политики, если ``policy`` не передана.
        base_score_noise: шум base ranker-а относительно истины.
        reward_model_noise: шум q_hat для Doubly Robust.
        policy / evaluation_policy: готовый бандит участника 1.
        max_age_days, min_impressions, boost_multiplier: Cold-Item Booster.
        time_to_n_threshold: N показов для метрики раскрутки.
        compute_cold_metrics: если False, не гоняет online-прогоны
            (нужно только OPE-валидации, где достаточно true_value).
    """

    eval_hint = evaluation_policy if evaluation_policy is not None else policy
    n_actions = int(getattr(eval_hint, "n_actions", N_ACTIONS))
    feature_dim = int(getattr(eval_hint, "context_dim", N_CONTEXT_FEATURES))

    rng = np.random.default_rng(seed)
    world = build_world(
        rng,
        n_actions=n_actions,
        feature_dim=feature_dim,
        n_days=n_days,
        base_score_noise=base_score_noise,
        cold_penalty=cold_penalty,
        target_ctr=target_ctr,
        seed=seed,
    )
    eval_policy = _evaluation_policy(seed, n_actions, epsilon, policy, evaluation_policy)
    if eval_policy.n_actions != world.catalog.n_actions:
        raise ValueError(
            f"policy.n_actions={eval_policy.n_actions} != catalog {world.catalog.n_actions}"
        )
    if getattr(eval_policy, "context_dim", feature_dim) != feature_dim:
        raise ValueError("policy.context_dim does not match simulator features")

    booster = ColdItemBooster(
        max_age_days=max_age_days,
        min_impressions=min_impressions,
        boost_multiplier=boost_multiplier,
    )
    batch = generate_batch(world, n_rounds, rng, max_age_days=max_age_days)
    logs, chosen, behavior_pscore, reward = _logging_pass(
        world,
        batch,
        booster,
        rng,
        logging_epsilon=logging_epsilon,
        seed=seed,
    )

    eval_dist = _policy_distribution_matrix(eval_policy, batch)
    rows = np.arange(n_rounds)
    evaluation_pscore = eval_dist[rows, chosen]
    true_value = _expected_ctr(eval_dist, batch.true_p)

    q_noise = rng.normal(scale=reward_model_noise, size=batch.true_logits.shape)
    q_hat_all = sigmoid(batch.true_logits + q_noise)
    q_hat_logged = q_hat_all[rows, chosen]
    q_hat_policy = np.sum(eval_dist * q_hat_all, axis=1)

    policy_name = eval_policy.name
    if hasattr(eval_policy, "epsilon"):
        policy_name = f"{eval_policy.name}(eps={float(eval_policy.epsilon):g})"

    ope_input = OPEInput(
        reward=reward,
        behavior_pscore=behavior_pscore,
        evaluation_pscore=evaluation_pscore,
        q_hat_logged=q_hat_logged,
        q_hat_policy=q_hat_policy,
        policy_name=policy_name,
    )

    new_ids = world.catalog.new_item_ids()
    online_eval = None
    eval_times = None
    base_times = None
    speedup = None
    cost = None
    exploration_share = (
        float(eval_policy.epsilon) if hasattr(eval_policy, "epsilon") else None
    )
    served_ctr = None
    baseline_served_ctr = None
    reached_eval = None
    reached_base = None
    if compute_cold_metrics:
        serving_booster = _serving_booster(eval_policy, booster)
        online_eval = run_online(
            world,
            batch,
            eval_policy,
            serving_booster,
            np.random.default_rng(seed + 91),
        )
        online_base = run_online(
            world,
            batch,
            _greedy_policy(n_actions, seed + 3),
            booster=None,
            rng=np.random.default_rng(seed + 92),
        )
        eval_times = time_to_n(
            online_eval["chosen"], new_ids, time_to_n_threshold, n_rounds
        )
        base_times = time_to_n(
            online_base["chosen"], new_ids, time_to_n_threshold, n_rounds
        )
        if new_ids.size:
            speedup = float(cold_item_speedup(base_times, eval_times))
            reached_eval = reached_n_share(eval_times, n_rounds)
            reached_base = reached_n_share(base_times, n_rounds)
        served_ctr = float(online_eval["expected_click"].mean())
        baseline_served_ctr = float(online_base["expected_click"].mean())
        cost = float(baseline_served_ctr - served_ctr)
        exploration_share = float(online_eval["exploration_share"])
    else:
        baseline_dist = _policy_distribution_matrix(
            _greedy_policy(n_actions, seed), batch
        )
        cost = float(_expected_ctr(baseline_dist, batch.true_p) - true_value)

    return SimulatorReplication(
        ope_input=ope_input,
        true_value=true_value,
        logs=logs,
        exploration_share=exploration_share,
        cold_item_speedup=speedup,
        ctr_cost=cost,
        metadata={
            "n_actions": n_actions,
            "n_rounds": n_rounds,
            "n_days": n_days,
            "n_warm_items": world.catalog.n_warm,
            "n_new_items": int(new_ids.size),
            "ope_true_ctr": true_value,
            "served_ctr": served_ctr,
            "baseline_served_ctr": baseline_served_ctr,
            "online_ctr": None if online_eval is None else online_eval["online_ctr"],
            "cold_item_impressions": (
                None if online_eval is None else int(online_eval["is_cold"].sum())
            ),
            "mean_time_to_n_eval": (
                None if eval_times is None or new_ids.size == 0 else float(eval_times.mean())
            ),
            "mean_time_to_n_baseline": (
                None if base_times is None or new_ids.size == 0 else float(base_times.mean())
            ),
            "reached_n_eval": reached_eval,
            "reached_n_baseline": reached_base,
            "logging_epsilon": logging_epsilon,
            "cold_penalty": cold_penalty,
            "reward_intercept": world.reward.intercept,
            "logging_policy": LOGGING_POLICY_NAME,
        },
    )
