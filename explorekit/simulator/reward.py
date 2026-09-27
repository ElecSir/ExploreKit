"""Скрытая истинная вероятность клика.

Политики и base ranker эту функцию не вызывают. Она нужна, чтобы:
1. сэмплировать reward 0/1 в симуляции;
2. точно посчитать true CTR любой политики (без онлайн-эксперимента).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from explorekit.simulator.items import ItemCatalog
from explorekit.simulator.users import UserGenerator

__all__ = ["TrueReward", "sigmoid", "calibrate_intercept"]


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Численно устойчивый sigmoid."""

    z_arr = np.asarray(z, dtype=float)
    scalar = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    exp_z = np.exp(z_arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    if scalar:
        return float(out[0])
    return out


@dataclass
class TrueReward:
    """P(click | user, item) = sigmoid(u·i + affinity[segment, category] + intercept)."""

    item_features: np.ndarray
    category_ids: np.ndarray
    affinity: np.ndarray
    intercept: float

    def logits(
        self,
        user_features: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Логиты для всех айтемов.

        Args:
            user_features: ``(d,)`` или ``(n, d)``.
            segment_ids: ``()`` / ``(n,)``. Если None — affinity не добавляется.

        Returns:
            ``(n_actions,)`` или ``(n, n_actions)``.
        """

        x = np.asarray(user_features, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        logits = x @ self.item_features.T + float(self.intercept)
        if segment_ids is not None:
            seg = np.asarray(segment_ids, dtype=int)
            if seg.ndim == 0:
                seg = np.full(len(x), int(seg))
            logits = logits + self.affinity[seg][:, self.category_ids]
        return logits[0] if single else logits

    def probabilities(
        self,
        user_features: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """P(click) в шкале [0, 1]."""

        return sigmoid(self.logits(user_features, segment_ids))

    def sample_click(
        self,
        rng: np.random.Generator,
        user_features: np.ndarray,
        item_id: int,
        segment_id: int | None = None,
    ) -> tuple[int, float]:
        """Сэмпл клика по одному показанному айтему. Возвращает (reward, p_true)."""

        p_all = self.probabilities(user_features, segment_id)
        p = float(p_all[int(item_id)])
        reward = int(rng.binomial(1, p))
        return reward, p


def calibrate_intercept(
    catalog: ItemCatalog,
    users: UserGenerator,
    affinity: np.ndarray,
    rng: np.random.Generator,
    *,
    target_ctr: float = 0.05,
    n_sample: int = 256,
) -> float:
    """Подбирает intercept так, чтобы CTR жадной политики на тёплых айтемах
    был около ``target_ctr`` (продуктовый порядок 3–8%, не среднее по каталогу).
    """

    if not 0.0 < target_ctr < 1.0:
        raise ValueError("target_ctr must be in (0, 1)")

    features, segment_ids, _ = users.sample_batch(n_sample, rng)
    warm = np.arange(catalog.n_warm)
    dots = features @ catalog.features[warm].T
    bonus = affinity[segment_ids][:, catalog.category_ids[warm]]
    base = dots + bonus

    lo, hi = -12.0, 4.0
    intercept = -3.0
    for _ in range(40):
        intercept = 0.5 * (lo + hi)
        greedy_ctr = float(sigmoid(base + intercept).max(axis=1).mean())
        if greedy_ctr < target_ctr:
            lo = intercept
        else:
            hi = intercept
    return float(intercept)
