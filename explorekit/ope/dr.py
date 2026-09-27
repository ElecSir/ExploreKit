"""Doubly Robust + reward model (задача 4).

    V_DR = (1/n) * sum_i [ E_{a~pi_e}[q_hat(x_i,a)]  +  w_i * (r_i - q_hat(x_i,a_i)) ]
             \\_______ Direct Method ________/   \\____ IPS-поправка на ошибку ____/

Свойство "двойной робастности": оценка остаётся несмещённой, если верно
ХОТЯ БЫ ОДНО из двух — либо propensity корректны, либо reward model хорошо
приближает истинное вознаграждение. На практике важнее другое: IPS-часть
взвешивает не всю награду, а только ОСТАТОК ``r - q_hat``, который в среднем
близок к нулю. Поэтому вклад больших весов гасится и дисперсия DR обычно
заметно ниже, чем у чистого IPS.

По ТЗ для прототипа достаточно LogisticRegression — нейросеть не нужна.

Ссылки: Dudik, Langford & Li, "Doubly Robust Policy Evaluation and
Learning", ICML 2011; в контексте рекомендаций — Jeunen & Goethals,
"An Empirical Evaluation of Doubly Robust Learning for Recommendation",
REVEAL@RecSys 2020.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from explorekit.ope.base import OPEEstimator, OPEInput
from explorekit.ope.ips import clip_weights

__all__ = ["DREstimator", "RewardModel"]


class DREstimator(OPEEstimator):
    """Doubly Robust с настраиваемым клиппингом весов.

    Требует, чтобы в ``OPEInput`` были заполнены ``q_hat_logged`` и
    ``q_hat_policy`` — их производит :class:`RewardModel` (или любая другая
    модель, лишь бы формат совпадал).
    """

    def __init__(self, max_weight: float | None = None) -> None:
        self.max_weight = max_weight
        self.name = "DR" if max_weight is None else f"DR(clip={max_weight:g})"

    def _check_requirements(self, data: OPEInput) -> None:
        if not data.has_reward_model:
            raise ValueError(
                "DREstimator требует q_hat_logged и q_hat_policy в OPEInput. "
                "Передайте q_hat_all_actions в build_ope_input(...) — "
                "их можно получить из RewardModel.predict_all_actions(...)."
            )

    def _estimate(self, data: OPEInput) -> float:
        w = clip_weights(data.importance_weights(), self.max_weight)
        direct = np.mean(data.q_hat_policy)
        correction = np.mean(w * (data.reward - data.q_hat_logged))
        return float(direct + correction)


class RewardModel:
    """q_hat(x, a) — вероятность клика по паре (контекст, действие).

    Единая модель на все действия: признаки = ``[контекст, признаки действия]``.
    Это нужно, чтобы модель обобщалась на редко показанные items — отдельная
    модель на каждый item развалилась бы на холодных айтемах, а именно они и
    интересуют ExploreKit.

    Обучать модель НАДО на отдельной части логов (см. ``split_for_reward_model``),
    иначе q_hat подгонится под те же данные, на которых потом считается DR,
    и оценка окажется оптимистично смещённой.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> None:
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        self._fitted = False
        self._n_actions: int | None = None

    @staticmethod
    def _design_matrix(context: np.ndarray, action_features: np.ndarray) -> np.ndarray:
        return np.concatenate([context, action_features], axis=1)

    def fit(
        self,
        context: np.ndarray,
        action_features_chosen: np.ndarray,
        reward: np.ndarray,
        n_actions: int,
    ) -> RewardModel:
        """Обучает модель на логах.

        Args:
            context: ``(n, d_context)`` признаки раунда.
            action_features_chosen: ``(n, d_action)`` признаки ПОКАЗАННОГО действия.
            reward: ``(n,)`` бинарная награда.
            n_actions: размер пространства действий.
        """
        X = self._design_matrix(context, action_features_chosen)
        if len(np.unique(reward)) < 2:
            raise ValueError(
                "в обучающей части логов только один класс наград — "
                "reward model обучить нельзя (увеличьте объём данных)"
            )
        self.model.fit(X, reward)
        self._fitted = True
        self._n_actions = n_actions
        return self

    def predict_all_actions(
        self, context: np.ndarray, action_features: np.ndarray
    ) -> np.ndarray:
        """Предсказывает q_hat(x, a) для КАЖДОГО действия.

        Args:
            context: ``(n, d_context)``.
            action_features: ``(n, n_actions, d_action)`` — признаки каждого
                действия-кандидата в каждом раунде.

        Returns:
            ``(n, n_actions)`` вероятностей клика.
        """
        if not self._fitted:
            raise RuntimeError("RewardModel не обучена — сначала вызовите fit(...)")
        n, n_actions, _ = action_features.shape
        out = np.zeros((n, n_actions), dtype=float)
        for a in range(n_actions):
            X = self._design_matrix(context, action_features[:, a, :])
            out[:, a] = self.model.predict_proba(X)[:, 1]
        return out


def split_for_reward_model(
    n_rounds: int, train_fraction: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Делит индексы логов на часть для обучения q_hat и часть для оценки.

    Returns:
        ``(train_idx, eval_idx)`` — непересекающиеся массивы индексов.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction должен лежать строго между 0 и 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rounds)
    n_train = int(n_rounds * train_fraction)
    return perm[:n_train], perm[n_train:]
