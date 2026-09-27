"""Тесты bootstrap-доверительных интервалов."""

from __future__ import annotations

import numpy as np
import pytest

from explorekit.ope import IPSEstimator, OPEInput, bootstrap_ci


class TestBootstrapMechanics:
    def test_point_estimate_uses_full_sample(self) -> None:
        values = np.arange(100, dtype=float)
        point, _, _ = bootstrap_ci(lambda idx: values[idx].mean(), n_rounds=100, seed=0)
        assert point == pytest.approx(values.mean())

    def test_ci_brackets_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=1000)
        point, lo, hi = bootstrap_ci(
            lambda idx: values[idx].mean(), n_rounds=1000, n_bootstrap=500, seed=0
        )
        assert lo <= point <= hi

    def test_reproducible_with_same_seed(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(size=500)
        fn = lambda idx: values[idx].mean()              
        first = bootstrap_ci(fn, 500, n_bootstrap=200, seed=42)
        second = bootstrap_ci(fn, 500, n_bootstrap=200, seed=42)
        assert first == second

    def test_different_seeds_give_different_ci(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(size=500)
        fn = lambda idx: values[idx].mean()              
        _, lo_a, _ = bootstrap_ci(fn, 500, n_bootstrap=200, seed=1)
        _, lo_b, _ = bootstrap_ci(fn, 500, n_bootstrap=200, seed=2)
        assert lo_a != lo_b

    def test_wider_confidence_level_gives_wider_interval(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=1000)
        fn = lambda idx: values[idx].mean()              
        _, lo95, hi95 = bootstrap_ci(
            fn, 1000, n_bootstrap=500, confidence_level=0.95, seed=0
        )
        _, lo99, hi99 = bootstrap_ci(
            fn, 1000, n_bootstrap=500, confidence_level=0.99, seed=0
        )
        assert (hi99 - lo99) > (hi95 - lo95)


class TestBootstrapValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_rounds": 0},
            {"n_rounds": 100, "n_bootstrap": 0},
            {"n_rounds": 100, "confidence_level": 1.5},
            {"n_rounds": 100, "confidence_level": 0.0},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(lambda idx: 0.0, **kwargs)


class TestCoverage:
    """Самая содержательная проверка: 95% CI должен накрывать истину
    примерно в 95% случаев. Если тут провал — интервалам нельзя верить."""

    def test_ci_covers_true_mean_at_nominal_rate(self) -> None:
        true_mean = 0.3
        n_trials = 200
        covered = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 + trial)
            sample = rng.binomial(1, true_mean, size=2000).astype(float)
            _, lo, hi = bootstrap_ci(
                lambda idx, sample=sample: sample[idx].mean(),
                n_rounds=2000,
                n_bootstrap=200,
                confidence_level=0.95,
                seed=trial,
            )
            covered += lo <= true_mean <= hi
        coverage = covered / n_trials
                                                           
        assert 0.90 <= coverage <= 0.99, f"coverage = {coverage:.1%}"

    def test_ips_ci_covers_known_policy_value(self) -> None:
        """То же, но уже для полноценного IPS на известной истине."""
        n_actions = 4
        true_ctr = np.array([0.1, 0.2, 0.3, 0.4])
        behavior_dist = np.array([0.25, 0.25, 0.25, 0.25])
        evaluation_dist = np.array([0.1, 0.2, 0.3, 0.4])
        true_value = float(evaluation_dist @ true_ctr)

        covered = 0
        n_trials = 100
        for trial in range(n_trials):
            rng = np.random.default_rng(2000 + trial)
            n = 5000
            actions = rng.choice(n_actions, size=n, p=behavior_dist)
            reward = rng.binomial(1, true_ctr[actions]).astype(float)
            data = OPEInput(
                reward=reward,
                behavior_pscore=behavior_dist[actions],
                evaluation_pscore=evaluation_dist[actions],
            )
            result = IPSEstimator().estimate_with_ci(data, n_bootstrap=200, seed=trial)
            covered += bool(result.covers(true_value))
        coverage = covered / n_trials
        assert coverage >= 0.88, f"coverage = {coverage:.1%}"
