"""Тесты диагностики надёжности."""

from __future__ import annotations

import numpy as np
import pytest

from explorekit.ope import OPEInput, ReliabilityStatus, compute_reliability


def make_input(behavior: np.ndarray, evaluation: np.ndarray) -> OPEInput:
    n = len(behavior)
    return OPEInput(
        reward=np.zeros(n),
        behavior_pscore=behavior,
        evaluation_pscore=evaluation,
        policy_name="test",
    )


class TestESS:
    def test_ess_equals_n_when_weights_are_uniform(self) -> None:
        """Все веса одинаковы -> оценка опирается на все наблюдения."""
        n = 1000
        pscore = np.full(n, 0.5)
        report = compute_reliability(make_input(pscore, pscore.copy()))
        assert report.ess == pytest.approx(n)
        assert report.ess_ratio == pytest.approx(1.0)

    def test_ess_collapses_when_one_weight_dominates(self) -> None:
        """Один гигантский вес -> ESS обваливается на порядки ниже N,
        сколько бы строк ни было в логах."""
        n = 1000
        behavior = np.full(n, 0.5)
        evaluation = np.full(n, 0.001)
        evaluation[0] = 0.99                                   
        report = compute_reliability(make_input(behavior, evaluation))
        assert report.ess < 0.01 * n
        assert report.status is ReliabilityStatus.UNRELIABLE

    def test_ess_never_exceeds_n(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        behavior = rng.uniform(0.05, 0.5, size=n)
        evaluation = rng.uniform(0.05, 0.5, size=n)
        report = compute_reliability(make_input(behavior, evaluation))
        assert report.ess <= n + 1e-9


class TestStatus:
    def test_identical_policies_are_reliable(self) -> None:
        n = 1000
        pscore = np.full(n, 0.3)
        report = compute_reliability(make_input(pscore, pscore.copy()))
        assert report.status is ReliabilityStatus.RELIABLE
        assert report.reasons == []
        assert "не выявила проблем" in report.explanation

    def test_moderate_mismatch_triggers_caution(self) -> None:
        rng = np.random.default_rng(1)
        n = 2000
        behavior = rng.uniform(0.1, 0.5, size=n)
                                                                        
        evaluation = behavior * rng.uniform(0.2, 3.0, size=n)
        report = compute_reliability(
            make_input(behavior, evaluation),
            ess_ratio_caution=0.95,
            ess_ratio_unreliable=0.01,
        )
        assert report.status is ReliabilityStatus.CAUTION
        assert report.reasons

    def test_unreliable_status_explains_reason(self) -> None:
        n = 1000
        behavior = np.full(n, 0.5)
        evaluation = np.full(n, 0.0001)
        evaluation[:2] = 0.99
        report = compute_reliability(make_input(behavior, evaluation))
        assert report.status is ReliabilityStatus.UNRELIABLE
        assert "ESS/N" in report.explanation

    def test_status_is_serialisable_for_dashboard(self) -> None:
        n = 100
        pscore = np.full(n, 0.5)
        payload = compute_reliability(make_input(pscore, pscore.copy())).to_dict()
        assert payload["status"] == "Reliable"
        assert set(payload) >= {
            "status",
            "explanation",
            "ess",
            "ess_ratio",
            "max_weight",
            "mean_weight",
            "fraction_clipped",
            "support_violations",
        }


class TestClippingFraction:
    def test_fraction_clipped_is_zero_without_max_weight(self) -> None:
        n = 100
        behavior = np.full(n, 0.01)
        evaluation = np.full(n, 0.9)
        report = compute_reliability(make_input(behavior, evaluation), max_weight=None)
        assert report.fraction_clipped == 0.0

    def test_fraction_clipped_counts_exceeding_weights(self) -> None:
        behavior = np.array([0.5, 0.5, 0.5, 0.5])
        evaluation = np.array([0.5, 0.5, 5.0, 5.0])                     
        report = compute_reliability(make_input(behavior, evaluation), max_weight=5.0)
        assert report.fraction_clipped == pytest.approx(0.5)

    def test_heavy_clipping_triggers_caution(self) -> None:
        n = 100
        behavior = np.full(n, 0.01)
        evaluation = np.full(n, 0.5)                                     
        report = compute_reliability(make_input(behavior, evaluation), max_weight=5.0)
        assert report.status is not ReliabilityStatus.RELIABLE
        assert any("Обрезано" in r for r in report.reasons)


class TestWeightStats:
    def test_max_and_mean_weight_reported(self) -> None:
        behavior = np.array([0.5, 0.5, 0.1])
        evaluation = np.array([0.5, 1.0, 0.5])                
        report = compute_reliability(make_input(behavior, evaluation))
        assert report.max_weight == pytest.approx(5.0)
        assert report.mean_weight == pytest.approx(8.0 / 3.0)

    def test_n_rounds_recorded(self) -> None:
        n = 77
        pscore = np.full(n, 0.5)
        report = compute_reliability(make_input(pscore, pscore.copy()))
        assert report.n_rounds == n


class TestVarianceDiagnostics:
    """Диагностика дисперсии (требование архитектуры: «клиппинг весов
    и диагностика дисперсии»)."""

    def test_identical_policies_have_zero_weight_variance(self) -> None:
        n = 500
        pscore = np.full(n, 0.4)
        report = compute_reliability(make_input(pscore, pscore.copy()))
        assert report.weight_variance == pytest.approx(0.0, abs=1e-12)
        assert report.weight_cv == pytest.approx(0.0, abs=1e-12)

    def test_cv_grows_when_policies_diverge(self) -> None:
        rng = np.random.default_rng(11)
        n = 2000
        behavior = np.full(n, 0.5)
        close = behavior * rng.uniform(0.9, 1.1, size=n)
        far = behavior * rng.uniform(0.01, 10.0, size=n)

        cv_close = compute_reliability(make_input(behavior, close)).weight_cv
        cv_far = compute_reliability(make_input(behavior, far)).weight_cv
        assert cv_far > cv_close

    def test_p99_is_below_max_weight(self) -> None:
        rng = np.random.default_rng(12)
        n = 1000
        behavior = rng.uniform(0.05, 0.5, size=n)
        evaluation = rng.uniform(0.05, 0.5, size=n)
        report = compute_reliability(make_input(behavior, evaluation))
        assert report.weight_p99 <= report.max_weight

    def test_ips_standard_error_matches_manual_formula(self) -> None:
        rng = np.random.default_rng(13)
        n = 1000
        behavior = rng.uniform(0.1, 0.5, size=n)
        evaluation = rng.uniform(0.1, 0.5, size=n)
        reward = rng.binomial(1, 0.3, size=n).astype(float)
        data = OPEInput(
            reward=reward,
            behavior_pscore=behavior,
            evaluation_pscore=evaluation,
        )
        report = compute_reliability(data)
        expected = np.std((evaluation / behavior) * reward, ddof=1) / np.sqrt(n)
        assert report.ips_standard_error == pytest.approx(expected)

    def test_standard_error_shrinks_with_more_data(self) -> None:
        """se ~ 1/sqrt(n): вчетверо больше данных -> примерно вдвое меньше se."""

        def se_for(n: int) -> float:
            rng = np.random.default_rng(14)
            behavior = np.full(n, 0.25)
            evaluation = np.full(n, 0.25)
            reward = rng.binomial(1, 0.3, size=n).astype(float)
            return compute_reliability(
                OPEInput(
                    reward=reward,
                    behavior_pscore=behavior,
                    evaluation_pscore=evaluation,
                )
            ).ips_standard_error

        assert se_for(2000) == pytest.approx(se_for(8000) * 2, rel=0.2)
