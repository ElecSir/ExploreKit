"""Базовые контракты OPE-модуля.

Здесь живёт единственная точка стыковки участника 2 с остальной командой:

    DecisionLog (участник 4)  +  evaluation policy (участник 1)
                          ↓
                       OPEInput
                          ↓
            IPS / SNIPS / DR  →  OPEResult

Всё остальное внутри ``explorekit.ope`` работает только с ``OPEInput`` и
ничего не знает ни про симулятор, ни про формат Parquet, ни про конкретные
классы политик. Это позволяет тестировать OPE до того, как будут готовы
модули участников 1, 3 и 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

__all__ = [
    "OPEInput",
    "OPEResult",
    "OPEEstimator",
    "ActionDistributionFn",
    "build_ope_input",
]


class ActionDistributionFn(Protocol):
    """Интерфейс evaluation policy со стороны участника 1.

    Должна вернуть массив ``(n_rounds, n_actions)`` вероятностей показа —
    ровно то, что ``BasePolicy.action_distribution(...)`` отдаёт для батча.
    """

    def __call__(self, context: np.ndarray, base_scores: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class OPEInput:
    """Всё, что нужно любому оценщику, в одном объекте.

    Attributes:
        reward: фактические награды из логов, шкала [0, 1].
        behavior_pscore: p_log(a_i | x_i) — propensity логирующей политики.
        evaluation_pscore: pi_e(a_i | x_i) — propensity ОЦЕНИВАЕМОЙ политики
            для того же действия, которое реально было показано.
        q_hat_logged: q_hat(x_i, a_i) — предсказание reward model для
            показанного действия. Нужно только для DR, для IPS/SNIPS ``None``.
        q_hat_policy: E_{a~pi_e}[q_hat(x_i, a)] — ожидание reward model под
            оцениваемой политикой. Нужно только для DR.
        policy_name: имя оцениваемой политики (для отчётов).
    """

    reward: np.ndarray
    behavior_pscore: np.ndarray
    evaluation_pscore: np.ndarray
    q_hat_logged: np.ndarray | None = None
    q_hat_policy: np.ndarray | None = None
    policy_name: str = "unnamed"

    def __post_init__(self) -> None:
        n = len(self.reward)
        for name in ("behavior_pscore", "evaluation_pscore"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name}: длина {len(arr)} != длине reward {n}")
        for name in ("q_hat_logged", "q_hat_policy"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name}: длина {len(arr)} != длине reward {n}")
        if np.any(self.behavior_pscore <= 0):
            raise ValueError(
                "behavior_pscore содержит нули или отрицательные значения — "
                "OPE математически некорректен при p_log(a|x) = 0 "
                "(нарушение common support). Проверьте валидатор логов."
            )

    def __len__(self) -> int:
        return len(self.reward)

    @property
    def has_reward_model(self) -> bool:
        return self.q_hat_logged is not None and self.q_hat_policy is not None

    def subset(self, idx: np.ndarray) -> OPEInput:
        """Подвыборка по индексам — используется bootstrap-ом."""
        return OPEInput(
            reward=self.reward[idx],
            behavior_pscore=self.behavior_pscore[idx],
            evaluation_pscore=self.evaluation_pscore[idx],
            q_hat_logged=None if self.q_hat_logged is None else self.q_hat_logged[idx],
            q_hat_policy=None if self.q_hat_policy is None else self.q_hat_policy[idx],
            policy_name=self.policy_name,
        )

    def importance_weights(self) -> np.ndarray:
        """w_i = pi_e(a_i|x_i) / p_log(a_i|x_i)."""
        return self.evaluation_pscore / self.behavior_pscore


@dataclass
class OPEResult:
    """Результат одной оценки. CTR хранится в шкале [0, 1] (общее правило №5)."""

    estimator_name: str
    policy_name: str
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    confidence_level: float = 0.95
    n_rounds: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def ci_width(self) -> float | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_high - self.ci_low

    def covers(self, true_value: float) -> bool | None:
        """Попадает ли истинное значение в доверительный интервал.

        Используется в synthetic validation для подсчёта CI coverage.
        """
        if self.ci_low is None or self.ci_high is None:
            return None
        return bool(self.ci_low <= true_value <= self.ci_high)

    def format_ctr(self) -> str:
        """Форматирование для dashboard: 0.031 -> '3.10%' (общее правило №5)."""
        if self.ci_low is None:
            return f"{self.estimate * 100:.2f}%"
        return (
            f"{self.estimate * 100:.2f}% "
            f"[{self.ci_low * 100:.2f}%, {self.ci_high * 100:.2f}%]"
        )


class OPEEstimator(ABC):
    """Единый интерфейс всех оценщиков (задача 1).

    Наследники обязаны реализовать только ``_estimate``. Bootstrap-интервалы,
    валидация входа и упаковка в ``OPEResult`` уже реализованы здесь, чтобы
    IPS/SNIPS/DR не дублировали один и тот же код.
    """

    name: str = "base"

    @abstractmethod
    def _estimate(self, data: OPEInput) -> float:
        """Точечная оценка ожидаемой награды политики на шкале [0, 1]."""

    def estimate(self, data: OPEInput) -> float:
        self._check_requirements(data)
        return self._estimate(data)

    def _check_requirements(self, data: OPEInput) -> None:
        """Переопределяется в DR, которому нужна reward model."""
        return None

    def estimate_with_ci(
        self,
        data: OPEInput,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 0,
    ) -> OPEResult:
        """Точечная оценка + bootstrap-доверительный интервал."""
        from explorekit.ope.bootstrap import bootstrap_ci

        self._check_requirements(data)
        point, lo, hi = bootstrap_ci(
            lambda idx: self._estimate(data.subset(idx)),
            n_rounds=len(data),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        return OPEResult(
            estimator_name=self.name,
            policy_name=data.policy_name,
            estimate=point,
            ci_low=lo,
            ci_high=hi,
            confidence_level=confidence_level,
            n_rounds=len(data),
        )


def build_ope_input(
    logs,
    evaluation_action_dist: np.ndarray,
    policy_name: str = "evaluation_policy",
    q_hat_all_actions: np.ndarray | None = None,
) -> OPEInput:
    """Адаптер: DecisionLog (участник 4) -> OPEInput.

    Args:
        logs: ``pandas.DataFrame`` в формате DecisionLog. Обязательные
            колонки: ``chosen_item``, ``propensity``, ``reward``.
            ``chosen_item`` должен быть индексом действия в диапазоне
            ``[0, n_actions)`` — если у участника 4 там строковые id,
            маппинг делается ДО вызова этой функции.
        evaluation_action_dist: ``(n_rounds, n_actions)`` — распределение
            оцениваемой политики, полученное от ``BasePolicy.action_distribution``.
        policy_name: имя оцениваемой политики.
        q_hat_all_actions: ``(n_rounds, n_actions)`` предсказаний reward model.
            Если ``None``, DR работать не сможет, IPS/SNIPS — смогут.

    Returns:
        Готовый ``OPEInput``.
    """
    required = {"chosen_item", "propensity", "reward"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"в DecisionLog не хватает колонок: {sorted(missing)}")

    n = len(logs)
    if evaluation_action_dist.shape[0] != n:
        raise ValueError(
            f"evaluation_action_dist: {evaluation_action_dist.shape[0]} строк "
            f"против {n} строк логов"
        )

    chosen = logs["chosen_item"].to_numpy(dtype=int)
    rows = np.arange(n)
    evaluation_pscore = evaluation_action_dist[rows, chosen]

    q_hat_logged = q_hat_policy = None
    if q_hat_all_actions is not None:
        q_hat_logged = q_hat_all_actions[rows, chosen]
        q_hat_policy = np.sum(evaluation_action_dist * q_hat_all_actions, axis=1)

    return OPEInput(
        reward=logs["reward"].to_numpy(dtype=float),
        behavior_pscore=logs["propensity"].to_numpy(dtype=float),
        evaluation_pscore=evaluation_pscore,
        q_hat_logged=q_hat_logged,
        q_hat_policy=q_hat_policy,
        policy_name=policy_name,
    )
