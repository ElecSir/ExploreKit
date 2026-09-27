"""Тесты Cold-Item Booster."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.cold_item_experiment import run_cold_item_experiment
from explorekit.cold_start import ColdItemBooster
from explorekit.policies import EpsilonGreedyPolicy
from explorekit.simulator import make_replication


def test_age_and_impressions_mark_cold_items() -> None:
    booster = ColdItemBooster(max_age_days=7, min_impressions=20, boost_multiplier=2.0)
    age = np.array([30, 2, 10, -5])
    impressions = np.array([100, 0, 5, 0])
    available = np.array([True, True, True, False])
    cold = booster.is_cold(age, impressions, available)
    assert cold.tolist() == [False, True, True, False]


def test_boost_raises_cold_item_scores() -> None:
    booster = ColdItemBooster(max_age_days=7, min_impressions=20, boost_multiplier=np.e)
    scores = np.array([0.0, 0.0, 1.0])
    decision = booster.apply(
        scores,
        age_days=np.array([30, 1, 30]),
        impressions=np.array([50, 0, 50]),
        available_mask=np.array([True, True, True]),
    )
    assert decision.cold_indices.tolist() == [1]
    assert decision.boosted_scores[1] == pytest.approx(1.0)
    assert decision.boosted_scores[0] == pytest.approx(0.0)


def test_epsilon_greedy_can_use_booster_subset() -> None:
    policy = EpsilonGreedyPolicy(
        n_actions=5, epsilon=1.0, min_epsilon=0.0, max_epsilon=1.0, seed=0
    )
    policy.exploration_subset = np.array([1, 2])
    dist = policy.action_distribution(
        context=np.zeros(3),
        base_scores=np.array([9.0, 0.1, 0.2, 0.3, 0.4]),
        candidate_items=np.arange(5),
    )
    assert dist[[0, 3, 4]].sum() == pytest.approx(0.0)
    assert dist[[1, 2]].sum() == pytest.approx(1.0)


def test_exploration_shows_new_items_faster() -> None:
    greedy = EpsilonGreedyPolicy(
        n_actions=16, epsilon=0.0, min_epsilon=0.0, max_epsilon=1.0, seed=0
    )
    aggressive = EpsilonGreedyPolicy(
        n_actions=16, epsilon=0.25, min_epsilon=0.0, max_epsilon=1.0, seed=0
    )
    base = make_replication(
        seed=21, n_rounds=320, policy=greedy, time_to_n_threshold=8
    )
    expl = make_replication(
        seed=21, n_rounds=320, policy=aggressive, time_to_n_threshold=8
    )
    assert expl.metadata["cold_item_impressions"] >= base.metadata["cold_item_impressions"]
    assert expl.cold_item_speedup is not None
    assert expl.cold_item_speedup >= 1.0


def test_cold_item_experiment_reads_epsilon_from_yaml() -> None:
    frame = run_cold_item_experiment(n_rounds=120)
    assert list(frame["preset"]) == [
        "baseline",
        "conservative",
        "moderate",
        "aggressive",
    ]
    assert frame["epsilon"].tolist() == pytest.approx([0.0, 0.05, 0.10, 0.25])
    assert frame["true_ctr"].notna().all()
    assert frame["speedup"].notna().all()
    assert frame.loc[frame["preset"] == "baseline", "ctr_cost"].iloc[0] == pytest.approx(
        0.0, abs=1e-6
    )
    assert frame.loc[frame["preset"] == "baseline", "speedup"].iloc[0] == pytest.approx(
        1.0, abs=1e-9
    )
