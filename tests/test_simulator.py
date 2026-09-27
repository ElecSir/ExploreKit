"""Тесты мира симулятора и контракта make_replication."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explorekit.logging import validate_logs
from explorekit.ope import DREstimator, IPSEstimator, SNIPSEstimator
from explorekit.ope.validation import run_single_comparison
from explorekit.policies import EpsilonGreedyPolicy
from explorekit.simulator import (
    ItemCatalog,
    UserGenerator,
    build_world,
    make_replication,
    sigmoid,
)
from explorekit.simulator.items import catalog_layout
from explorekit.simulator.metrics import cold_item_speedup, time_to_n
from explorekit.simulator.reward import calibrate_intercept


def test_sigmoid_bounds() -> None:
    values = sigmoid(np.array([-20.0, 0.0, 20.0]))
    assert values[0] == pytest.approx(0.0, abs=1e-8)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0, abs=1e-8)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


def test_catalog_uses_int_item_ids() -> None:
    rng = np.random.default_rng(0)
    catalog = ItemCatalog(n_actions=12, feature_dim=4, rng=rng)
    assert catalog.n_actions == 12
    assert all(isinstance(item.item_id, int) for item in catalog.items)
    assert catalog.available_at(0).tolist() == list(range(catalog.n_warm))
    new_ids = catalog.new_item_ids()
    assert new_ids.size >= 2
    assert np.all(catalog.created_at_day[new_ids] > 0)
    later = catalog.available_at(10)
    assert later.size == 12


def test_catalog_layout_covers_all_slots() -> None:
    n_warm, waves = catalog_layout(80)
    assert n_warm + sum(count for _, count in waves) == 80


def test_click_probabilities_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    world = build_world(rng, n_actions=16, feature_dim=4, seed=1)
    users = world.users.sample_batch(32, rng)
    probs = world.reward.probabilities(users[0], users[1])
    assert probs.shape == (32, 16)
    assert np.all(probs > 0.0)
    assert np.all(probs < 1.0)


def test_intercept_calibration_near_target() -> None:
    rng = np.random.default_rng(2)
    catalog = ItemCatalog(n_actions=20, feature_dim=4, rng=rng)
    users = UserGenerator(feature_dim=4, rng=rng)
    affinity = rng.normal(scale=0.3, size=(users.n_segments, catalog.n_categories))
    intercept = calibrate_intercept(
        catalog, users, affinity, np.random.default_rng(3), target_ctr=0.05
    )
    features, seg, _ = users.sample_batch(400, np.random.default_rng(4))
    from explorekit.simulator.reward import TrueReward

    reward = TrueReward(catalog.features, catalog.category_ids, affinity, intercept)
    warm_p = reward.probabilities(features, seg)[:, : catalog.n_warm]
    greedy_ctr = float(warm_p.max(axis=1).mean())
    assert 0.03 < greedy_ctr < 0.08


def _small_policy(n_actions: int = 12, epsilon: float = 0.1, seed: int = 0):
    return EpsilonGreedyPolicy(
        n_actions=n_actions,
        epsilon=epsilon,
        min_epsilon=0.0,
        max_epsilon=1.0,
        seed=seed,
    )


def test_make_replication_contract() -> None:
    policy = _small_policy()
    rep = make_replication(seed=42, n_rounds=240, policy=policy, time_to_n_threshold=8)
    assert hasattr(rep, "ope_input")
    assert hasattr(rep, "true_value")
    assert np.isfinite(rep.true_value)
    assert 0.0 < rep.true_value < 1.0
    assert len(rep.ope_input) == 240
    assert np.all(rep.ope_input.behavior_pscore > 0.0)
    assert np.all(np.isfinite(rep.ope_input.reward))
    assert set(np.unique(rep.ope_input.reward)).issubset({0.0, 1.0})
    assert isinstance(rep.logs[0].chosen_item, (int, np.integer))
    assert 0 <= int(rep.logs[0].chosen_item) < policy.n_actions
    assert not isinstance(rep.logs[0].chosen_item, str)
    assert rep.cold_item_speedup is not None
    assert np.isfinite(rep.cold_item_speedup)
    assert rep.ctr_cost is not None


def test_logs_pass_validator() -> None:
    rep = make_replication(seed=7, n_rounds=80, policy=_small_policy(seed=7))
    frame = pd.DataFrame([row.to_record() for row in rep.logs])
    report = validate_logs(frame, raise_on_error=False)
    assert report.valid, report.errors


def test_make_replication_is_reproducible() -> None:
    policy_a = _small_policy(seed=11)
    policy_b = _small_policy(seed=11)
    first = make_replication(seed=11, n_rounds=120, policy=policy_a)
    second = make_replication(seed=11, n_rounds=120, policy=policy_b)
    assert first.true_value == pytest.approx(second.true_value)
    assert np.allclose(first.ope_input.reward, second.ope_input.reward)
    assert np.allclose(first.ope_input.behavior_pscore, second.ope_input.behavior_pscore)


def test_ope_path_can_skip_online_metrics() -> None:
    rep = make_replication(
        seed=3,
        n_rounds=60,
        policy=_small_policy(seed=3),
        compute_cold_metrics=False,
    )
    assert np.isfinite(rep.true_value)
    assert len(rep.ope_input) == 60
    assert rep.cold_item_speedup is None
    assert rep.exploration_share == pytest.approx(0.1)


def test_polina_ope_can_consume_replication() -> None:
    rep = make_replication(seed=5, n_rounds=180, policy=_small_policy(seed=5))
    table = run_single_comparison(
        rep,
        [IPSEstimator(max_weight=15.0), SNIPSEstimator(max_weight=15.0), DREstimator(max_weight=15.0)],
        n_bootstrap=30,
        seed=5,
    )
    assert len(table) == 3
    assert np.all(np.isfinite(table["estimate"]))


def test_time_to_n_and_speedup() -> None:
    chosen = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    times = time_to_n(chosen, np.array([1, 2]), n_impressions=3, n_rounds=8)
    assert times.tolist() == [4, 7]
    assert cold_item_speedup(np.array([10.0, 20.0]), np.array([5.0, 10.0])) == pytest.approx(2.0)
    assert cold_item_speedup(np.array([100.0, 10.0]), np.array([25.0, 10.0])) == pytest.approx(2.5)
