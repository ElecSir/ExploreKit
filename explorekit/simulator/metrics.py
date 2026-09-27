"""Метрики exploration / cold-start для синтетических прогонов."""

from __future__ import annotations

import numpy as np

__all__ = ["ctr", "ctr_cost", "time_to_n", "reached_n_share", "cold_item_speedup"]


def ctr(rewards: np.ndarray) -> float:
    """clicks / impressions на шкале [0, 1]."""

    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        raise ValueError("rewards must be non-empty")
    return float(rewards.mean())


def ctr_cost(baseline_ctr: float, exploration_ctr: float) -> float:
    """Потеря CTR относительно baseline. Хранится как разница на шкале [0, 1]."""

    return float(baseline_ctr - exploration_ctr)


def time_to_n(
    chosen_items: np.ndarray,
    new_item_ids: np.ndarray,
    n_impressions: int,
    n_rounds: int | None = None,
) -> np.ndarray:
    """Для каждого нового item — шаг, на котором он набрал ``n_impressions``.

    Если к концу прогона порог не взят, пишем ``n_rounds`` (цензурирование).
    Возвращает массив той же длины, что ``new_item_ids``.
    """

    if n_impressions <= 0:
        raise ValueError("n_impressions must be positive")
    chosen = np.asarray(chosen_items, dtype=int)
    horizon = int(chosen.size if n_rounds is None else n_rounds)
    new_ids = np.asarray(new_item_ids, dtype=int)
    times = np.full(new_ids.shape, horizon, dtype=int)
    if new_ids.size == 0 or chosen.size == 0:
        return times

    counts = {int(i): 0 for i in new_ids.tolist()}
    remaining = set(counts)
    for t, item in enumerate(chosen, start=1):
        item_id = int(item)
        if item_id not in remaining:
            continue
        counts[item_id] += 1
        if counts[item_id] >= n_impressions:
            idx = int(np.flatnonzero(new_ids == item_id)[0])
            times[idx] = t
            remaining.remove(item_id)
            if not remaining:
                break
    return times


def reached_n_share(times: np.ndarray, n_rounds: int) -> float:
    """Доля новых items, которые набрали N показов до конца прогона."""

    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return 0.0
    return float(np.mean(times < float(n_rounds)))


def cold_item_speedup(
    baseline_times: np.ndarray,
    exploration_times: np.ndarray,
) -> float:
    """Среднее по айтемам отношение baseline_time / exploration_time.

    Считать mean(base) / mean(expl) нельзя: цензурированные айтемы
    (время = n_rounds) затягивают оба средних и сжимают speedup к 1.
    """

    base = np.asarray(baseline_times, dtype=float)
    expl = np.asarray(exploration_times, dtype=float)
    if base.shape != expl.shape or base.size == 0:
        raise ValueError("time-to-N arrays must be non-empty and aligned")
    if np.any(expl <= 0):
        raise ValueError("exploration time-to-N must be positive")
    return float(np.mean(base / expl))
