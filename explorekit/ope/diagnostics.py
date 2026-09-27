"""Диагностика надёжности OPE-оценки (задача 6).

Оценка OPE может быть формально посчитана всегда — даже когда доверять ей
нельзя. Этот модуль отвечает на вопрос "а можно ли показывать это число
клиенту" и выдаёт статус ``Reliable`` / ``Use with caution`` / ``Unreliable``
вместе с человекочитаемым объяснением причины.

Ключевая метрика — Effective Sample Size:

    ESS = (sum_i w_i)^2 / sum_i w_i^2

Смысл: сколько "эффективных" независимых наблюдений реально несёт информацию
после взвешивания. Если все веса равны, ESS = N. Если оценка держится на
десятке показов с огромными весами, ESS будет ~10, сколько бы миллионов
строк ни лежало в логах — и доверительный интервал, посчитанный как будто по
миллиону строк, будет обманчиво узким.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from explorekit.ope.base import OPEInput

__all__ = ["ReliabilityStatus", "ReliabilityReport", "compute_reliability"]


class ReliabilityStatus(StrEnum):
    RELIABLE = "Reliable"
    CAUTION = "Use with caution"
    UNRELIABLE = "Unreliable"


@dataclass
class ReliabilityReport:
    """Полный отчёт о надёжности оценки для одной политики."""

    status: ReliabilityStatus
    reasons: list[str] = field(default_factory=list)
    ess: float = 0.0
    ess_ratio: float = 0.0
    max_weight: float = 0.0
    mean_weight: float = 0.0
    weight_variance: float = 0.0
    weight_cv: float = 0.0
    weight_p99: float = 0.0
    ips_standard_error: float = 0.0
    fraction_clipped: float = 0.0
    support_violations: int = 0
    support_violation_rate: float = 0.0
    n_rounds: int = 0

    @property
    def explanation(self) -> str:
        """Одна строка для dashboard и логов."""
        if not self.reasons:
            return (
                "Диагностика не выявила проблем: веса умеренные, "
                "перекрытие политик достаточное."
            )
        return " ".join(self.reasons)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "explanation": self.explanation,
            "ess": self.ess,
            "ess_ratio": self.ess_ratio,
            "max_weight": self.max_weight,
            "mean_weight": self.mean_weight,
            "weight_variance": self.weight_variance,
            "weight_cv": self.weight_cv,
            "weight_p99": self.weight_p99,
            "ips_standard_error": self.ips_standard_error,
            "fraction_clipped": self.fraction_clipped,
            "support_violations": self.support_violations,
            "support_violation_rate": self.support_violation_rate,
            "n_rounds": self.n_rounds,
        }


def compute_reliability(
    data: OPEInput,
    max_weight: float | None = None,
    ess_ratio_caution: float = 0.20,
    ess_ratio_unreliable: float = 0.05,
    clipped_fraction_caution: float = 0.05,
    support_violation_caution: float = 0.01,
) -> ReliabilityReport:
    """Считает диагностики и выносит вердикт.

    Args:
        data: вход OPE.
        max_weight: тот же порог клиппинга, что передан в оценщик — нужен,
            чтобы посчитать долю обрезанных весов.
        ess_ratio_caution: ниже этого ESS/N — предупреждение.
        ess_ratio_unreliable: ниже этого ESS/N — оценка непригодна.
        clipped_fraction_caution: доля обрезанных весов, выше которой
            смещение от клиппинга становится существенным.
        support_violation_caution: доля нарушений common support, выше
            которой оценка систематически занижена.

    Returns:
        :class:`ReliabilityReport`.
    """
    w = data.importance_weights()
    n = len(w)

    sum_sq = float(np.sum(w**2))
    ess = float((np.sum(w) ** 2) / sum_sq) if sum_sq > 0 else 0.0
    ess_ratio = ess / n if n > 0 else 0.0

                                   
                                                                      
                                                                     
                                                                       
                                                                  
    weight_variance = float(np.var(w, ddof=1)) if n > 1 else 0.0
    mean_weight = float(w.mean()) if n > 0 else 0.0
    weight_cv = float(np.sqrt(weight_variance) / mean_weight) if mean_weight > 0 else 0.0
    weight_p99 = float(np.quantile(w, 0.99)) if n > 0 else 0.0
                                                                      
                                                                     
                                                                  
    ips_terms = w * data.reward
    ips_standard_error = float(np.std(ips_terms, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    fraction_clipped = float(np.mean(w > max_weight)) if max_weight is not None else 0.0

                                                                             
                                                                     
                                                                        
                                                             
    support_violations = int(
        np.sum((data.evaluation_pscore > 0) & (data.behavior_pscore <= 1e-10))
    )
    support_violation_rate = support_violations / n if n > 0 else 0.0

    reasons: list[str] = []
    status = ReliabilityStatus.RELIABLE

    if ess_ratio < ess_ratio_unreliable:
        status = ReliabilityStatus.UNRELIABLE
        reasons.append(
            f"ESS/N = {ess_ratio:.1%} — оценка фактически держится на "
            f"{ess:.0f} эффективных наблюдениях из {n}; политики почти не перекрываются."
        )
    elif ess_ratio < ess_ratio_caution:
        status = ReliabilityStatus.CAUTION
        reasons.append(
            f"ESS/N = {ess_ratio:.1%} — перекрытие политик слабое, "
            f"доверительный интервал может быть занижен."
        )

    if support_violation_rate > support_violation_caution:
        status = ReliabilityStatus.UNRELIABLE
        reasons.append(
            f"Нарушение common support в {support_violation_rate:.1%} раундов: "
            f"оцениваемая политика показывает действия, которых нет в логах — "
            f"оценка систематически занижена."
        )

    if fraction_clipped > clipped_fraction_caution:
        if status is ReliabilityStatus.RELIABLE:
            status = ReliabilityStatus.CAUTION
        reasons.append(
            f"Обрезано {fraction_clipped:.1%} весов (max_weight={max_weight:g}) — "
            f"клиппинг вносит заметное смещение вниз."
        )

    return ReliabilityReport(
        status=status,
        reasons=reasons,
        ess=ess,
        ess_ratio=ess_ratio,
        max_weight=float(w.max()) if n > 0 else 0.0,
        mean_weight=mean_weight,
        weight_variance=weight_variance,
        weight_cv=weight_cv,
        weight_p99=weight_p99,
        ips_standard_error=ips_standard_error,
        fraction_clipped=fraction_clipped,
        support_violations=support_violations,
        support_violation_rate=support_violation_rate,
        n_rounds=n,
    )
