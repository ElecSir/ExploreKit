"""Adapter from experiment config to a simulator `make_replication` provider."""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from explorekit.ope import OPEInput
from explorekit.policies import BasePolicy

__all__ = ["IntegratedReplication", "load_make_replication", "make_replication"]


@dataclass
class IntegratedReplication:
    """Normalized superset of ``Replication(ope_input, true_value)``."""

    ope_input: OPEInput
    true_value: float
    exploration_share: float | None = None
    cold_item_speedup: float | None = None
    ctr_cost: float | None = None
    logs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_make_replication(module_name: str) -> Callable[..., Any]:
    """Import a simulator provider exposing the agreed function."""

    module = importlib.import_module(module_name)
    fn = getattr(module, "make_replication", None)
    if not callable(fn):
        raise AttributeError(
            f"{module_name} must expose make_replication(seed) -> Replication"
        )
    return fn


def _supported_kwargs(
    fn: Callable[..., Any],
    config: dict[str, Any],
    policy: BasePolicy,
) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    simulation = config.get("simulation", {})
    cold = config.get("cold_start", {})
    candidates: dict[str, Any] = {
        "n_rounds": simulation.get("n_rounds"),
        "base_score_noise": simulation.get("base_score_noise"),
        "reward_model_noise": simulation.get("reward_model_noise"),
        "n_days": simulation.get("n_days"),
        "logging_epsilon": simulation.get("logging_epsilon"),
        "target_ctr": simulation.get("target_ctr"),
        "cold_penalty": simulation.get("cold_penalty"),
        "max_age_days": cold.get("max_age_days"),
        "min_impressions": cold.get("min_impressions"),
        "boost_multiplier": cold.get("boost_multiplier"),
        "time_to_n_threshold": cold.get("time_to_n"),
    }
    if hasattr(policy, "epsilon"):
        candidates["epsilon"] = float(getattr(policy, "epsilon"))
    if "policy" in params:
        candidates["policy"] = policy
    if "evaluation_policy" in params:
        candidates["evaluation_policy"] = policy

    return {
        name: value
        for name, value in candidates.items()
        if name in params and value is not None
    }


def _metric(raw: Any, name: str) -> Any:
    if hasattr(raw, name):
        return getattr(raw, name)
    metadata = getattr(raw, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(name)
    return None


def make_replication(
    config: dict[str, Any],
    policy: BasePolicy,
    *,
    seed: int | None = None,
) -> IntegratedReplication:
    """Call a simulator provider while preserving the minimum agreed contract."""

    module_name = str(config["simulation"]["provider"])
    fn = load_make_replication(module_name)
    resolved_seed = int(config["seed"] if seed is None else seed)
    raw = fn(resolved_seed, **_supported_kwargs(fn, config, policy))
    if not hasattr(raw, "ope_input") or not hasattr(raw, "true_value"):
        raise TypeError(
            "simulator result must expose ope_input and true_value attributes"
        )
    if not isinstance(raw.ope_input, OPEInput):
        raise TypeError("replication.ope_input must be explorekit.ope.OPEInput")

    metadata = dict(getattr(raw, "metadata", {}) or {})
    metadata.setdefault("simulation_provider", module_name)
    return IntegratedReplication(
        ope_input=raw.ope_input,
        true_value=float(raw.true_value),
        exploration_share=_metric(raw, "exploration_share"),
        cold_item_speedup=_metric(raw, "cold_item_speedup"),
        ctr_cost=_metric(raw, "ctr_cost"),
        logs=_metric(raw, "logs"),
        metadata=metadata,
    )
