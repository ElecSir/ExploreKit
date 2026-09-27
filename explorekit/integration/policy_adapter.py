"""Integration with the real participant-1 policy factory."""
from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from explorekit.policies import BasePolicy, build_active_policy, load_policies_config

from explorekit.integration.config import project_path

__all__ = ["build_active_policy_for_experiment", "public_policy_params"]


def _provider_dimensions(config: dict[str, Any]) -> tuple[int | None, int | None]:
    module_name = config.get("simulation", {}).get("provider")
    if not module_name:
        return None, None
    module = importlib.import_module(module_name)
    n_actions = getattr(module, "N_ACTIONS", None)
    context_dim = getattr(module, "N_CONTEXT_FEATURES", None)
    return (
        None if n_actions is None else int(n_actions),
        None if context_dim is None else int(context_dim),
    )


def build_active_policy_for_experiment(config: dict[str, Any]) -> BasePolicy:
    """Load ``configs/policies.yaml``, apply preset overrides and call the factory."""

    policy_cfg = config["policy"]
    config_path = project_path(policy_cfg["config_path"])
    factory_cfg = deepcopy(load_policies_config(config_path))
    active = str(policy_cfg["active_policy"])
    if active not in factory_cfg["policies"]:
        raise KeyError(
            f"active policy {active!r} not found in {Path(config_path).name}"
        )

    factory_cfg["active_policy"] = active
    factory_cfg["seed"] = int(config["seed"])
    factory_cfg["policies"][active].update(policy_cfg.get("overrides", {}))

    n_actions, context_dim = _provider_dimensions(config)
    active_spec = factory_cfg["policies"][active]
    if n_actions is not None and "n_actions" in active_spec:
        active_spec["n_actions"] = n_actions
    if context_dim is not None and "context_dim" in active_spec:
        active_spec["context_dim"] = context_dim

    return build_active_policy(factory_cfg)


def public_policy_params(policy: BasePolicy) -> dict[str, Any]:
    """Return stable scalar/list parameters suitable for DecisionLog metadata."""

    excluded = {"rng", "A", "b", "_last_context"}
    params: dict[str, Any] = {}
    for key, value in vars(policy).items():
        if key in excluded or key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            params[key] = value
        elif hasattr(value, "tolist"):
            params[key] = value.tolist()
    return params
