from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Decision:
    """Одно решение политики на одном шаге.

    Инварианты (проверяются в __post_init__):
      * chosen_item — int-индекс в [0, n_actions), НЕ строковый item_id;
      * propensity == probabilities[chosen_item] и > 0;
      * sum(probabilities) == 1.
    """
    chosen_item: int
    propensity: float
    probabilities: np.ndarray
    is_exploration: bool
    policy_name: str = "base"

    def __post_init__(self) -> None:
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if not isinstance(self.chosen_item, (int, np.integer)):
            raise TypeError(
                f"chosen_item must be int index, got {type(self.chosen_item).__name__}. "
                "Mapping string item_id -> int index should happen in the simulator."
            )
        if not 0 <= self.chosen_item < len(self.probabilities):
            raise ValueError("chosen_item out of [0, n_actions)")
        if not np.isclose(self.probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError(f"probabilities must sum to 1, got {self.probabilities.sum()}")
        if not np.isclose(self.propensity, self.probabilities[self.chosen_item]):
            raise ValueError("propensity must equal probabilities[chosen_item]")
        if self.propensity <= 0:
            raise ValueError("propensity must be > 0 (common support violated)")


class BasePolicy(ABC):
    """Общий интерфейс всех bandit-политик.

    Все политики работают в едином пространстве действий [0, n_actions).
    Кандидаты передаются как int-массив индексов. base_scores — массив
    длины n_actions со скорами от базового ранкера.
    """

    name: str = "base"

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def action_distribution(
        self,
        context: np.ndarray,                                                    
        base_scores: np.ndarray,                                              
        candidate_items: np.ndarray | None = None,                
    ) -> np.ndarray:
        """Возвращает (n_rounds, n_actions). Если вход 1D — возвращает 1D."""
        ...

    @abstractmethod
    def select_action(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> Decision:
        ...

    @abstractmethod
    def update(
        self,
        decision: Decision,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        ...
