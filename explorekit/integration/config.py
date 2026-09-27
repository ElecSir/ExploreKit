"""Loading and validating participant-4 experiment configs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "project_path", "validate_config"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    """Resolve config/data paths relative to the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _PROJECT_ROOT / candidate


def _read_yaml(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {resolved}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one preset and inherit the participant-2 OPE config."""

    config_path = project_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config root must be a mapping")

    ope_ref = dict(raw.get("ope", {}))
    ope_config_path = ope_ref.pop("config_path", "configs/ope.yaml")
    base_ope = _read_yaml(ope_config_path)

    merged = deepcopy(raw)
    merged["ope"] = _deep_merge(base_ope.get("ope", {}), ope_ref)
    for section in (
        "bootstrap",
        "reward_model",
        "diagnostics",
        "synthetic_validation",
        "open_bandit",
    ):
        merged[section] = _deep_merge(
            base_ope.get(section, {}), raw.get(section, {})
        )

    merged.setdefault("seed", int(base_ope.get("seed", 42)))
    merged["_config_path"] = str(config_path)
    merged["_ope_config_path"] = str(project_path(ope_config_path))
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate fields required by the end-to-end runner."""

    required = [
        "preset_name",
        "seed",
        "experiment",
        "policy",
        "simulation",
        "cold_start",
        "ope",
        "bootstrap",
        "diagnostics",
        "output",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config is missing keys: {missing}")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be int")

    policy_cfg = config["policy"]
    if "config_path" not in policy_cfg or "active_policy" not in policy_cfg:
        raise ValueError("policy requires config_path and active_policy")
    epsilon = policy_cfg.get("overrides", {}).get("epsilon")
    if epsilon is not None and not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("policy.overrides.epsilon must be in [0, 1]")

    n_rounds = int(config["simulation"].get("n_rounds", 0))
    if n_rounds <= 0:
        raise ValueError("simulation.n_rounds must be positive")
    if not config["simulation"].get("provider"):
        raise ValueError("simulation.provider is required")

    max_weight = config["ope"].get("max_weight")
    if max_weight is not None and float(max_weight) <= 0:
        raise ValueError("ope.max_weight must be positive or null")
    if int(config["bootstrap"].get("n_bootstrap", 0)) <= 0:
        raise ValueError("bootstrap.n_bootstrap must be positive")
    confidence = float(config["bootstrap"].get("confidence_level", 0.0))
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap.confidence_level must be in (0, 1)")

    cold = config["cold_start"]
    if int(cold.get("max_age_days", 0)) < 0:
        raise ValueError("cold_start.max_age_days must be non-negative")
    if int(cold.get("min_impressions", 0)) < 0:
        raise ValueError("cold_start.min_impressions must be non-negative")
