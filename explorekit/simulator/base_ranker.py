"""Базовый ранкер поверх скрытой истины.

Это не отдельная recommender-система. Ранкер видит зашумлённую версию
true logit и дополнительно занижает скоры молодых айтемов — так в проде
модель «не доверяет» новинкам, пока по ним мало данных.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from explorekit.simulator.reward import TrueReward, sigmoid

__all__ = ["BaseRanker"]


@dataclass
class BaseRanker:
    """Noisy estimate of true reward, которым пользуется бандит."""

    noise_scale: float = 0.3
    cold_penalty: float = 1.2
    return_probabilities: bool = False

    def scores(
        self,
        true_logits: np.ndarray,
        *,
        age_days: np.ndarray,
        max_age_days: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Скоры длины ``n_actions`` (или батч ``(n, n_actions)``).

        Недоступные айтемы маскируются позже через ``candidate_items``,
        поэтому здесь считаем полный вектор. Молодые айтемы получают
        отрицательный сдвиг ``cold_penalty``.
        """

        logits = np.asarray(true_logits, dtype=float)
        noise = rng.normal(scale=self.noise_scale, size=logits.shape)
        ages = np.asarray(age_days, dtype=float)
        if logits.ndim == 2 and ages.ndim == 1:
            young = ages < float(max_age_days)
            penalty = np.where(young[None, :], self.cold_penalty, 0.0)
        else:
            penalty = np.where(ages < float(max_age_days), self.cold_penalty, 0.0)
        scores = logits + noise - penalty
        if self.return_probabilities:
            return sigmoid(scores)
        return scores
