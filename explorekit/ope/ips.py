"""Inverse Propensity Scoring (задача 2).

    V_IPS(pi_e) = (1/n) * sum_i [ w_i * r_i ],   w_i = pi_e(a_i|x_i) / p_log(a_i|x_i)

Несмещённая оценка ценности политики при двух условиях:
  1. propensity логирующей политики записаны верно;
  2. common support: p_log(a|x) > 0 везде, где pi_e(a|x) > 0.

Главная практическая проблема — дисперсия: если оцениваемая политика
концентрируется на действиях, которые логирующая показывала редко, отдельные
веса становятся огромными и оценка "держится" на единичных наблюдениях.
Отсюда clipping через ``max_weight`` (вносит отрицательное смещение, зато
резко снижает дисперсию — классический bias/variance trade-off).

Ссылка: Horvitz & Thompson (1952); в контексте рекомендаций —
Saito & Joachims, "Counterfactual Learning and Evaluation for Recommender
Systems", RecSys 2021 (tutorial).
"""

from __future__ import annotations

import numpy as np

from explorekit.ope.base import OPEEstimator, OPEInput

__all__ = ["IPSEstimator", "clip_weights"]


def clip_weights(weights: np.ndarray, max_weight: float | None) -> np.ndarray:
    """Обрезает importance weights сверху. ``None`` = без обрезки."""
    if max_weight is None:
        return weights
    if max_weight <= 0:
        raise ValueError("max_weight должен быть положительным или None")
    return np.minimum(weights, max_weight)


class IPSEstimator(OPEEstimator):
    """IPS с настраиваемым клиппингом весов.

    Args:
        max_weight: верхняя граница importance weight. ``None`` — не обрезать.
            Значение приходит из config (общее правило №4).
    """

    def __init__(self, max_weight: float | None = None) -> None:
        self.max_weight = max_weight
        self.name = "IPS" if max_weight is None else f"IPS(clip={max_weight:g})"

    def _estimate(self, data: OPEInput) -> float:
        w = clip_weights(data.importance_weights(), self.max_weight)
        return float(np.mean(w * data.reward))
