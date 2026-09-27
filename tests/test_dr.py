"""Тесты Doubly Robust и reward model."""

from __future__ import annotations

import numpy as np
import pytest

from explorekit.ope import (
    DREstimator,
    IPSEstimator,
    OPEInput,
    RewardModel,
    split_for_reward_model,
)


def make_dr_input(
    reward: np.ndarray,
    behavior: np.ndarray,
    evaluation: np.ndarray,
    q_logged: np.ndarray,
    q_policy: np.ndarray,
) -> OPEInput:
    return OPEInput(
        reward=reward,
        behavior_pscore=behavior,
        evaluation_pscore=evaluation,
        q_hat_logged=q_logged,
        q_hat_policy=q_policy,
        policy_name="test",
    )


class TestDRRequirements:
    def test_dr_without_reward_model_raises(self) -> None:
        data = OPEInput(
            reward=np.array([1.0, 0.0]),
            behavior_pscore=np.array([0.5, 0.5]),
            evaluation_pscore=np.array([0.5, 0.5]),
        )
        with pytest.raises(ValueError, match="q_hat_logged"):
            DREstimator().estimate(data)


class TestDRReducesToKnownCases:
    def test_dr_with_zero_reward_model_equals_ips(self) -> None:
        """Если q_hat ≡ 0, DR вырождается ровно в IPS."""
        rng = np.random.default_rng(0)
        n = 500
        behavior = rng.uniform(0.1, 0.9, size=n)
        evaluation = rng.uniform(0.1, 0.9, size=n)
        reward = rng.binomial(1, 0.3, size=n).astype(float)
        data = make_dr_input(reward, behavior, evaluation, np.zeros(n), np.zeros(n))

        assert DREstimator().estimate(data) == pytest.approx(
            IPSEstimator().estimate(data)
        )

    def test_dr_with_perfect_reward_model_and_zero_residual(self) -> None:
        """Если q_hat идеально предсказывает reward, IPS-поправка зануляется
        и DR равен чистому Direct Method."""
        rng = np.random.default_rng(1)
        n = 500
        behavior = rng.uniform(0.1, 0.9, size=n)
        evaluation = rng.uniform(0.1, 0.9, size=n)
        reward = rng.binomial(1, 0.4, size=n).astype(float)
        q_policy = rng.uniform(0.2, 0.6, size=n)
                                                                
        data = make_dr_input(reward, behavior, evaluation, reward.copy(), q_policy)

        assert DREstimator().estimate(data) == pytest.approx(q_policy.mean())


class TestDoubleRobustness:
    """Главное свойство DR: несмещённость сохраняется, если верна ХОТЯ БЫ
    ОДНА из двух частей. Проверяем обе ветки по отдельности."""

    @staticmethod
    def _simulate(seed: int, n: int = 200_000):
        rng = np.random.default_rng(seed)
        n_actions = 4
        true_ctr = np.array([0.1, 0.2, 0.3, 0.4])
        behavior_dist = np.array([0.4, 0.3, 0.2, 0.1])
        evaluation_dist = np.array([0.1, 0.2, 0.3, 0.4])

        actions = rng.choice(n_actions, size=n, p=behavior_dist)
        reward = rng.binomial(1, true_ctr[actions]).astype(float)
        true_value = float(evaluation_dist @ true_ctr)
        return actions, reward, behavior_dist, evaluation_dist, true_ctr, true_value

    def test_unbiased_when_reward_model_is_wrong(self) -> None:
        """propensity верны, q_hat — заведомо кривая константа.
        DR обязан всё равно попасть в истину."""
        actions, reward, b_dist, e_dist, _, true_value = self._simulate(10)
        n = len(reward)
        wrong_q = np.full(n, 0.9)                                  
        data = make_dr_input(
            reward, b_dist[actions], e_dist[actions], wrong_q, np.full(n, 0.9)
        )
        assert DREstimator().estimate(data) == pytest.approx(true_value, abs=0.01)

    def test_unbiased_when_propensities_are_wrong(self) -> None:
        """q_hat точная, а propensity испорчены. DR всё равно близок к истине,
        потому что остаток (reward - q_hat) в среднем нулевой."""
        actions, reward, b_dist, e_dist, true_ctr, true_value = self._simulate(11)
        n = len(reward)
        exact_q_logged = true_ctr[actions]
        exact_q_policy = np.full(n, float(e_dist @ true_ctr))

        corrupted_behavior = np.full(n, 0.25)                       
        data = make_dr_input(
            reward, corrupted_behavior, e_dist[actions], exact_q_logged, exact_q_policy
        )
        assert DREstimator().estimate(data) == pytest.approx(true_value, abs=0.01)


class TestRewardModel:
    def test_fit_and_predict_shapes(self) -> None:
        rng = np.random.default_rng(2)
        n, n_actions, d_ctx, d_act = 500, 5, 3, 2
        context = rng.normal(size=(n, d_ctx))
        action_features = rng.normal(size=(n, n_actions, d_act))
        chosen = rng.integers(0, n_actions, size=n)
        chosen_features = action_features[np.arange(n), chosen]
        reward = rng.binomial(1, 0.3, size=n).astype(float)

        model = RewardModel().fit(context, chosen_features, reward, n_actions)
        q = model.predict_all_actions(context, action_features)

        assert q.shape == (n, n_actions)
        assert np.all((q >= 0) & (q <= 1))

    def test_predict_before_fit_raises(self) -> None:
        model = RewardModel()
        with pytest.raises(RuntimeError, match="не обучена"):
            model.predict_all_actions(np.zeros((2, 3)), np.zeros((2, 4, 2)))

    def test_single_class_reward_raises(self) -> None:
        model = RewardModel()
        with pytest.raises(ValueError, match="один класс"):
            model.fit(np.zeros((10, 2)), np.zeros((10, 2)), np.zeros(10), 3)

    def test_model_learns_signal(self) -> None:
        """Если клик действительно зависит от признака действия, модель
        обязана это уловить, иначе DR теряет смысл."""
        rng = np.random.default_rng(4)
        n, n_actions = 4000, 3
        context = rng.normal(size=(n, 2))
        action_features = rng.normal(size=(n, n_actions, 1))
        chosen = rng.integers(0, n_actions, size=n)
        chosen_features = action_features[np.arange(n), chosen]
                                                  
        prob = 1 / (1 + np.exp(-3 * chosen_features[:, 0]))
        reward = rng.binomial(1, prob).astype(float)

        model = RewardModel().fit(context, chosen_features, reward, n_actions)
        high = model.predict_all_actions(context, np.full((n, n_actions, 1), 2.0))
        low = model.predict_all_actions(context, np.full((n, n_actions, 1), -2.0))
        assert high.mean() > low.mean() + 0.3


class TestSplit:
    def test_split_is_disjoint_and_complete(self) -> None:
        train, evaluate = split_for_reward_model(1000, train_fraction=0.5, seed=0)
        assert len(train) + len(evaluate) == 1000
        assert len(np.intersect1d(train, evaluate)) == 0

    def test_split_is_reproducible(self) -> None:
        a, _ = split_for_reward_model(500, seed=7)
        b, _ = split_for_reward_model(500, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            split_for_reward_model(100, train_fraction=1.5)
