"""Self-Normalized Inverse Propensity Scoring (задача 3).

    V_SNIPS(pi_e) = sum_i(w_i * r_i) / sum_i(w_i)

Отличие от IPS — в знаменателе стоит не n, а фактическая сумма весов.
Интуиция: у IPS сумма весов в конечной выборке почти никогда не равна n,
хотя в матожидании равна; SNIPS исправляет именно эту "случайную
разбалансировку" выборки. Цена — небольшое смещение (оценка перестаёт быть
строго несмещённой), выигрыш — заметно меньшая дисперсия, особенно когда
политики далеки друг от друга и веса сильно разбросаны.

Полезное свойство для sanity-check: SNIPS всегда лежит внутри диапазона
наблюдаемых наград, то есть для бинарного клика — в [0, 1]. IPS такой
гарантии не даёт и может выдать оценку CTR > 1 на "плохих" логах.

Ссылка: Swaminathan & Joachims, "The Self-Normalized Estimator for
Counterfactual Learning", NeurIPS 2015.
"""

from __future__ import annotations

import numpy as np

from explorekit.ope.base import OPEEstimator, OPEInput
from explorekit.ope.ips import clip_weights

__all__ = ["SNIPSEstimator"]


class SNIPSEstimator(OPEEstimator):
    """SNIPS с настраиваемым клиппингом весов.

    Args:
        max_weight: верхняя граница importance weight. ``None`` — не обрезать.
    """

    def __init__(self, max_weight: float | None = None) -> None:
        self.max_weight = max_weight
        self.name = "SNIPS" if max_weight is None else f"SNIPS(clip={max_weight:g})"

    def _estimate(self, data: OPEInput) -> float:
        w = clip_weights(data.importance_weights(), self.max_weight)
        denominator = float(np.sum(w))
        if denominator == 0.0:
                                                                               
            return 0.0
        return float(np.sum(w * data.reward) / denominator)
