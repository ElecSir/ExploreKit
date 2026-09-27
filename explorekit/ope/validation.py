"""Synthetic validation оценщиков (задача 7).

Логика проверки: участник 3 отдаёт нам симулятор, в котором ИЗВЕСТНА
истинная ценность политики (симулятор сам задал P(click | user, item), так
что может посчитать её точно). Мы же, имея на руках только логи, считаем
IPS/SNIPS/DR и смотрим, насколько близко подобрались.

Одного прогона мало: он покажет, что оценка "примерно похожа на правду", но
не отличит несмещённый оценщик с большим разбросом от смещённого со
стабильным. Поэтому гоняем 50–100 повторов с разными seed и считаем:

    Bias        = mean(estimate) - true_value    — систематическая ошибка
    Std         = std(estimate)                  — разброс между повторами
    RMSE        = sqrt(mean((estimate - true)^2)) — общее качество
    CI coverage = доля повторов, где 95% CI накрыл истину

CI coverage — самая важная строка отчёта и главный тест на честность
доверительных интервалов. Если он заявлен как 95%, а покрывает истину в
60% случаев, то интервал врёт, и клиенту его показывать нельзя.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from explorekit.ope.base import OPEEstimator, OPEInput

__all__ = [
    "Replication",
    "EstimatorValidationStats",
    "ValidationReport",
    "run_single_comparison",
    "run_validation",
]


@dataclass
class Replication:
    """Один прогон симулятора: логи + известная истинная ценность политики."""

    ope_input: OPEInput
    true_value: float


@dataclass
class EstimatorValidationStats:
    """Агрегированные метрики качества одного оценщика."""

    estimator_name: str
    n_replications: int
    true_value: float
    mean_estimate: float
    bias: float
    std: float
    rmse: float
    mean_abs_error: float
    ci_coverage: float
    mean_ci_width: float


@dataclass
class ValidationReport:
    """Итог валидации по всем оценщикам."""

    stats: list[EstimatorValidationStats] = field(default_factory=list)
    confidence_level: float = 0.95

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.__dict__ for s in self.stats])

    def format_table(self) -> str:
        header = (
            f"{'Оценщик':<16}{'Среднее':>10}{'Bias':>11}{'Std':>10}"
            f"{'RMSE':>10}{'CI cover':>11}{'CI width':>11}"
        )
        lines = [header, "-" * len(header)]
        for s in self.stats:
            lines.append(
                f"{s.estimator_name:<16}{s.mean_estimate:>10.4f}{s.bias:>+11.4f}"
                f"{s.std:>10.4f}{s.rmse:>10.4f}{s.ci_coverage:>10.1%}"
                f"{s.mean_ci_width:>11.4f}"
            )
        return "\n".join(lines)


def run_single_comparison(
    replication: Replication,
    estimators: list[OPEEstimator],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Одиночное сравнение True CTR vs оценки, с абсолютной ошибкой и 95% CI.

    Это таблица "для человека" — её удобно показывать в отчёте и на dashboard.
    """
    rows = []
    for est in estimators:
        result = est.estimate_with_ci(
            replication.ope_input,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        rows.append(
            {
                "estimator": result.estimator_name,
                "true_ctr": replication.true_value,
                "estimate": result.estimate,
                "abs_error": abs(result.estimate - replication.true_value),
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "covers_true": result.covers(replication.true_value),
            }
        )
    return pd.DataFrame(rows)


def run_validation(
    make_replication: Callable[[int], Replication],
    estimators: list[OPEEstimator],
    n_replications: int = 50,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ValidationReport:
    """Многократная валидация: Bias, Std, RMSE и CI coverage.

    Args:
        make_replication: функция от seed, возвращающая :class:`Replication`.
            Реализуется участником 3 (симулятор); на время разработки
            подменяется стабом из ``experiments/_stub_environment.py``.
        estimators: список оценщиков для сравнения.
        n_replications: сколько независимых прогонов симулятора (50–100 по ТЗ).
        n_bootstrap: bootstrap-повторов внутри каждого прогона. Можно меньше,
            чем в продовом отчёте: здесь важна не точность отдельного CI, а
            доля покрытий по многим прогонам.
        confidence_level: уровень доверия.
        seed: базовый seed; прогон ``i`` использует ``seed + i``.

    Returns:
        :class:`ValidationReport`.
    """
    if n_replications < 2:
        raise ValueError("нужно хотя бы 2 повтора, иначе Std не определён")

    estimates: dict[str, list[float]] = {est.name: [] for est in estimators}
    coverage: dict[str, list[bool]] = {est.name: [] for est in estimators}
    ci_widths: dict[str, list[float]] = {est.name: [] for est in estimators}
    true_values: list[float] = []

    for i in range(n_replications):
        rep = make_replication(seed + i)
        true_values.append(rep.true_value)
        for est in estimators:
            result = est.estimate_with_ci(
                rep.ope_input,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed + i,
            )
            estimates[est.name].append(result.estimate)
            coverage[est.name].append(bool(result.covers(rep.true_value)))
            ci_widths[est.name].append(float(result.ci_width))

    true_value = float(np.mean(true_values))
    report = ValidationReport(confidence_level=confidence_level)
    for est in estimators:
        values = np.asarray(estimates[est.name], dtype=float)
                                                                                
                                                                            
        errors = values - np.asarray(true_values, dtype=float)
        report.stats.append(
            EstimatorValidationStats(
                estimator_name=est.name,
                n_replications=n_replications,
                true_value=true_value,
                mean_estimate=float(values.mean()),
                bias=float(errors.mean()),
                std=float(values.std(ddof=1)),
                rmse=float(np.sqrt(np.mean(errors**2))),
                mean_abs_error=float(np.mean(np.abs(errors))),
                ci_coverage=float(np.mean(coverage[est.name])),
                mean_ci_width=float(np.mean(ci_widths[est.name])),
            )
        )
    return report
