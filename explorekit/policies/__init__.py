"""Bandit policies для ExploreKit.

Публичный API:
    Decision, BasePolicy           — базовый контракт
    EpsilonGreedyPolicy            — ε-greedy
    LinUCBPolicy                   — Linear UCB
    ThompsonSamplingPolicy         — Linear Thompson Sampling

    load_policies_config           — загрузка configs/policies.yaml
    build_policy                   — сборка политики по spec
    build_policy_by_name           — сборка по имени из YAML
    build_active_policy            — сборка активной политики
"""

from __future__ import annotations

from typing import Any

from .base import BasePolicy, Decision
from .config import load_policies_config
from .epsilon_greedy import EpsilonGreedyPolicy
from .linucb import LinUCBPolicy
from .thompson import ThompsonSamplingPolicy


__all__ = [
    "BasePolicy",
    "Decision",
    "EpsilonGreedyPolicy",
    "LinUCBPolicy",
    "ThompsonSamplingPolicy",
    "load_policies_config",
    "build_policy",
    "build_policy_by_name",
    "build_active_policy",
]


                                           
_POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    "epsilon_greedy": EpsilonGreedyPolicy,
    "linucb": LinUCBPolicy,
    "thompson": ThompsonSamplingPolicy,
}


def build_policy(spec: dict[str, Any], default_seed: int | None = None) -> BasePolicy:
    """Собирает политику из одного spec (секции из policies.yaml).

    spec — это словарь вида {"type": "linucb", "n_actions": 80, ...}.
    default_seed подставляется, если в spec нет своего seed
    (политики наследуют глобальный seed из конфига).
    """
    if "type" not in spec:
        raise ValueError(f"policy spec has no 'type' field: {spec}")

    params = dict(spec)
    ptype = params.pop("type")

    if ptype not in _POLICY_REGISTRY:
        raise ValueError(
            f"unknown policy type '{ptype}'; "
            f"known: {sorted(_POLICY_REGISTRY.keys())}"
        )

    if default_seed is not None and "seed" not in params:
        params["seed"] = default_seed

    return _POLICY_REGISTRY[ptype](**params)


def build_policy_by_name(name: str, cfg: dict | None = None) -> BasePolicy:
    """Собирает политику по имени секции из конфига.

    Пример:
        cfg = load_policies_config()
        policy = build_policy_by_name("linucb", cfg)
    """
    if cfg is None:
        cfg = load_policies_config()

    policies = cfg["policies"]
    if name not in policies:
        raise KeyError(
            f"policy '{name}' not found in config; "
            f"available: {sorted(policies.keys())}"
        )

    return build_policy(policies[name], default_seed=cfg.get("seed"))


def build_active_policy(cfg: dict | None = None) -> BasePolicy:
    """Собирает политику, помеченную в YAML как active_policy.

    Используется в боевом пайплайне (участник 4): клиент меняет
    одну строку в policies.yaml — политика меняется без правки кода.
    """
    if cfg is None:
        cfg = load_policies_config()

    active = cfg.get("active_policy")
    if active is None:
        raise ValueError("config has no 'active_policy' field")

    return build_policy_by_name(active, cfg)
