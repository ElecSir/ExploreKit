"""Bootstrap-доверительные интервалы (задача 5).

Один общий helper на все оценщики: percentile bootstrap по раундам.
Ресэмплим индексы логов с возвращением, пересчитываем оценку на каждой
псевдовыборке и берём эмпирические квантили — это корректно работает для
любого из наших оценщиков, потому что все они являются функциями
эмпирического распределения раундов.

Seed всегда приходит из config (общее правило №3): внутри используется
``np.random.default_rng(seed)``, никаких глобальных ``np.random.seed``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["bootstrap_ci"]


def bootstrap_ci(
    estimate_fn: Callable[[np.ndarray], float],
    n_rounds: int,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap.

    Args:
        estimate_fn: функция, считающая точечную оценку по индексам раундов.
            Должна принимать массив индексов (возможно, с повторами).
        n_rounds: число раундов в логах.
        n_bootstrap: число bootstrap-повторов.
        confidence_level: уровень доверия, например 0.95.
        seed: seed генератора (из config).

    Returns:
        ``(point_estimate, ci_low, ci_high)``.

    Raises:
        ValueError: при некорректных аргументах.
    """
    if n_rounds <= 0:
        raise ValueError("n_rounds должно быть положительным")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap должно быть положительным")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level должен лежать строго между 0 и 1")

    point = float(estimate_fn(np.arange(n_rounds)))

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_rounds, size=n_rounds)
        samples[b] = estimate_fn(idx)

    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
