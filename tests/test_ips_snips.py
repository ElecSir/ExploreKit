"""Тесты IPS и SNIPS."""

from __future__ import annotations

import numpy as np
import pytest

from explorekit.ope import IPSEstimator, OPEInput, SNIPSEstimator, clip_weights


def make_input(
    reward: np.ndarray,
    behavior: np.ndarray,
    evaluation: np.ndarray,
) -> OPEInput:
    return OPEInput(
        reward=reward,
        behavior_pscore=behavior,
        evaluation_pscore=evaluation,
        policy_name="test",
    )


class TestIdenticalPolicies:
    """Если оцениваемая политика совпадает с логирующей, оба оценщика
    обязаны вернуть ровно наблюдаемый средний reward: все веса равны 1."""

    def test_ips_equals_mean_reward(self) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        pscore = rng.uniform(0.05, 0.5, size=n)
        reward = rng.binomial(1, 0.3, size=n).astype(float)
        data = make_input(reward, pscore, pscore.copy())

        assert IPSEstimator().estimate(data) == pytest.approx(reward.mean())

    def test_snips_equals_mean_reward(self) -> None:
        rng = np.random.default_rng(1)
        n = 1000
        pscore = rng.uniform(0.05, 0.5, size=n)
        reward = rng.binomial(1, 0.3, size=n).astype(float)
        data = make_input(reward, pscore, pscore.copy())

        assert SNIPSEstimator().estimate(data) == pytest.approx(reward.mean())


class TestUnbiasedness:
    """IPS несмещён: на данных, где мы сами знаем истину, оценка сходится."""

    def test_ips_recovers_known_value(self) -> None:
        rng = np.random.default_rng(42)
        n = 200_000
        n_actions = 4
        true_ctr_per_action = np.array([0.1, 0.2, 0.3, 0.4])

        behavior_dist = np.array([0.4, 0.3, 0.2, 0.1])
        evaluation_dist = np.array([0.1, 0.2, 0.3, 0.4])

        actions = rng.choice(n_actions, size=n, p=behavior_dist)
        reward = rng.binomial(1, true_ctr_per_action[actions]).astype(float)
        data = make_input(reward, behavior_dist[actions], evaluation_dist[actions])

        true_value = float(evaluation_dist @ true_ctr_per_action)
        assert IPSEstimator().estimate(data) == pytest.approx(true_value, abs=0.005)


class TestDistantPolicies:
    """Далёкие политики: веса разлетаются, но оценщики обязаны оставаться
    в разумных пределах, а SNIPS — не выходить за диапазон наград."""

    def test_snips_stays_within_reward_range(self) -> None:
        rng = np.random.default_rng(7)
        n = 5000
        behavior = rng.uniform(0.001, 0.01, size=n)                                  
        evaluation = rng.uniform(0.5, 0.9, size=n)                                
        reward = rng.binomial(1, 0.2, size=n).astype(float)
        data = make_input(reward, behavior, evaluation)

        snips_value = SNIPSEstimator().estimate(data)
        assert 0.0 <= snips_value <= 1.0

    def test_ips_can_exceed_reward_range_without_clipping(self) -> None:
        """Демонстрация известной слабости IPS: без клиппинга оценка CTR
        может вылезти выше 1, что физически невозможно. Это не баг теста —
        это причина, по которой в конфиге есть max_weight."""
        n = 100
        behavior = np.full(n, 0.001)
        evaluation = np.full(n, 0.9)
        reward = np.ones(n)
        data = make_input(reward, behavior, evaluation)

        assert IPSEstimator().estimate(data) > 1.0
        assert SNIPSEstimator().estimate(data) == pytest.approx(1.0)


class TestClipping:
    def test_clip_weights_caps_values(self) -> None:
        w = np.array([0.5, 5.0, 50.0])
        clipped = clip_weights(w, max_weight=10.0)
        np.testing.assert_allclose(clipped, [0.5, 5.0, 10.0])

    def test_clip_none_is_identity(self) -> None:
        w = np.array([0.5, 5.0, 50.0])
        np.testing.assert_allclose(clip_weights(w, None), w)

    def test_clipping_reduces_ips_estimate(self) -> None:
        """Клиппинг вносит смещение вниз — это ожидаемое поведение."""
        n = 1000
        rng = np.random.default_rng(3)
        behavior = rng.uniform(0.001, 0.1, size=n)
        evaluation = rng.uniform(0.1, 0.5, size=n)
        reward = rng.binomial(1, 0.5, size=n).astype(float)
        data = make_input(reward, behavior, evaluation)

        unclipped = IPSEstimator().estimate(data)
        clipped = IPSEstimator(max_weight=5.0).estimate(data)
        assert clipped <= unclipped

    def test_invalid_max_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            clip_weights(np.array([1.0]), max_weight=0.0)


class TestEdgeCases:
    def test_zero_behavior_pscore_is_rejected(self) -> None:
        """Нулевой propensity логирующей политики делает OPE некорректным —
        должны падать громко, а не молча делить на ноль."""
        with pytest.raises(ValueError, match="common support"):
            OPEInput(
                reward=np.array([1.0, 0.0]),
                behavior_pscore=np.array([0.5, 0.0]),
                evaluation_pscore=np.array([0.5, 0.5]),
            )

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="длина"):
            OPEInput(
                reward=np.array([1.0, 0.0]),
                behavior_pscore=np.array([0.5]),
                evaluation_pscore=np.array([0.5, 0.5]),
            )

    def test_snips_handles_all_zero_evaluation_pscore(self) -> None:
        """Оцениваемая политика вообще не пересекается с логами —
        возвращаем 0.0, а не NaN."""
        n = 10
        data = make_input(
            reward=np.ones(n),
            behavior=np.full(n, 0.5),
            evaluation=np.zeros(n),
        )
        assert SNIPSEstimator().estimate(data) == 0.0

    def test_all_zero_rewards_give_zero_estimate(self) -> None:
        n = 100
        rng = np.random.default_rng(5)
        data = make_input(
            reward=np.zeros(n),
            behavior=rng.uniform(0.1, 0.9, size=n),
            evaluation=rng.uniform(0.1, 0.9, size=n),
        )
        assert IPSEstimator().estimate(data) == 0.0
        assert SNIPSEstimator().estimate(data) == 0.0
