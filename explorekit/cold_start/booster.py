"""Cold-Item Booster: кто считается новым и как поднимать шанс показа.

Не заменяет бандита. Для epsilon-greedy отдаёт ``exploration_subset`` —
Сергей уже размазывает ε только по этому подмножеству. Дополнительно
сдвигает base scores молодых айтемов на ``log(boost_multiplier)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ColdItemBooster", "ColdItemDecision"]


@dataclass(frozen=True)
class ColdItemDecision:
    """Результат бустера на одном шаге."""

    cold_mask: np.ndarray
    cold_indices: np.ndarray
    boosted_scores: np.ndarray

    @property
    def n_cold(self) -> int:
        return int(self.cold_indices.size)


class ColdItemBooster:
    """Помечает cold items по возрасту и/или числу показов."""

    def __init__(
        self,
        max_age_days: int = 7,
        min_impressions: int = 20,
        boost_multiplier: float = 2.0,
    ) -> None:
        if max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        if min_impressions < 0:
            raise ValueError("min_impressions must be non-negative")
        if boost_multiplier <= 0:
            raise ValueError("boost_multiplier must be positive")

        self.max_age_days = int(max_age_days)
        self.min_impressions = int(min_impressions)
        self.boost_multiplier = float(boost_multiplier)
        self._log_boost = float(np.log(self.boost_multiplier))

    def is_cold(
        self,
        age_days: np.ndarray,
        impressions: np.ndarray,
        available_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(n_actions,)`` bool: item cold и уже доступен в каталоге."""

        ages = np.asarray(age_days)
        imps = np.asarray(impressions)
        cold = (ages < self.max_age_days) | (imps < self.min_impressions)
        if available_mask is not None:
            cold = cold & np.asarray(available_mask, dtype=bool)
        else:
            # ещё не вышедший айтем имеет отрицательный возраст относительно
            # текущего дня только если created_at в будущем: его нельзя бустить.
            cold = cold & (ages >= 0)
        return cold

    def apply(
        self,
        base_scores: np.ndarray,
        age_days: np.ndarray,
        impressions: np.ndarray,
        available_mask: np.ndarray | None = None,
    ) -> ColdItemDecision:
        """Пометить cold items и поднять их base scores."""

        scores = np.array(base_scores, dtype=float, copy=True)
        cold_mask = self.is_cold(age_days, impressions, available_mask)
        if scores.ndim != 1:
            raise ValueError("apply() expects 1D base_scores for one round")
        if cold_mask.shape != scores.shape:
            raise ValueError("cold mask and base_scores length differ")
        scores[cold_mask] += self._log_boost
        return ColdItemDecision(
            cold_mask=cold_mask,
            cold_indices=np.flatnonzero(cold_mask),
            boosted_scores=scores,
        )
