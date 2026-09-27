"""Validation of the canonical ExploreKit decision log."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = ["LogValidationError", "LogValidationReport", "validate_logs"]


class LogValidationError(ValueError):
    """Raised when DecisionLog rows violate the shared contract."""


@dataclass
class LogValidationReport:
    """Result of validating one log batch."""

    n_rows: int
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise LogValidationError("; ".join(self.errors))


def _to_frame(logs: pd.DataFrame | Iterable[Any]) -> pd.DataFrame:
    if isinstance(logs, pd.DataFrame):
        return logs.copy()
    rows: list[dict[str, Any]] = []
    for row in logs:
        if hasattr(row, "to_record"):
            rows.append(row.to_record())
        elif isinstance(row, Mapping):
            rows.append(dict(row))
        else:
            rows.append(vars(row))
    return pd.DataFrame(rows)


def _numeric_vector(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _integer_vector(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not all(isinstance(x, (int, np.integer)) for x in arr.tolist()):
        return None
    return arr.astype(int)


def validate_logs(
    logs: pd.DataFrame | Iterable[Any],
    *,
    atol: float = 1e-6,
    raise_on_error: bool = True,
) -> LogValidationReport:
    """Validate DecisionLog invariants required by policy and OPE modules."""

    frame = _to_frame(logs)
    required = {
        "timestamp",
        "request_id",
        "user_segment",
        "context",
        "candidate_items",
        "base_scores",
        "chosen_item",
        "propensity",
        "probabilities",
        "policy_name",
        "policy_params",
        "reward",
        "is_exploration",
        "is_cold_item",
    }
    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"missing required columns: {missing}")
        report = LogValidationReport(len(frame), False, errors, warnings)
        if raise_on_error:
            report.raise_for_errors()
        return report

    if frame.empty:
        errors.append("DecisionLog is empty")

    propensity = pd.to_numeric(frame["propensity"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(propensity)):
        errors.append("propensity contains NaN/inf/non-numeric values")
    if np.any((propensity <= 0.0) | (propensity > 1.0)):
        errors.append("propensity must satisfy 0 < propensity <= 1")

    reward = pd.to_numeric(frame["reward"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(reward)):
        errors.append("reward contains NaN/inf/non-numeric values")
    if not np.all(np.isin(reward, [0.0, 1.0])):
        errors.append("reward must belong to {0, 1}")

    for idx, row in frame.iterrows():
        context = _numeric_vector(row["context"])
        scores = _numeric_vector(row["base_scores"])
        probs = _numeric_vector(row["probabilities"])
        candidates = _integer_vector(row["candidate_items"])

        if context is None:
            errors.append(f"row {idx}: context must be a finite numeric vector")
        if scores is None:
            errors.append(f"row {idx}: base_scores must be a finite numeric vector")
        if probs is None:
            errors.append(f"row {idx}: probabilities must be a finite numeric vector")
        if candidates is None:
            errors.append(f"row {idx}: candidate_items must be integer action indices")
        if scores is None or probs is None or candidates is None:
            continue

        n_actions = len(probs)
        if len(scores) != n_actions:
            errors.append(
                f"row {idx}: len(base_scores)={len(scores)} != "
                f"len(probabilities)={n_actions}"
            )
        if len(np.unique(candidates)) != len(candidates):
            errors.append(f"row {idx}: candidate_items contains duplicates")
        if np.any((candidates < 0) | (candidates >= n_actions)):
            errors.append(
                f"row {idx}: candidate_items must lie in [0, {n_actions})"
            )
        if np.any(probs < 0.0):
            errors.append(f"row {idx}: probabilities contains negative values")
        if not np.isclose(probs.sum(), 1.0, atol=atol):
            errors.append(
                f"row {idx}: probabilities sum is {probs.sum():.8f}, not 1"
            )

        chosen = row["chosen_item"]
        if not isinstance(chosen, (int, np.integer)):
            errors.append(
                f"row {idx}: chosen_item must be int action index, "
                f"got {type(chosen).__name__}"
            )
            continue
        chosen_idx = int(chosen)
        if not 0 <= chosen_idx < n_actions:
            errors.append(f"row {idx}: chosen_item out of [0, {n_actions})")
            continue
        if chosen_idx not in set(candidates.tolist()):
            errors.append(f"row {idx}: chosen_item is not in candidate_items")
        if not np.isclose(float(row["propensity"]), probs[chosen_idx], atol=atol):
            errors.append(
                f"row {idx}: propensity does not equal probabilities[chosen_item]"
            )

        if not isinstance(row["policy_params"], Mapping):
            errors.append(f"row {idx}: policy_params must be a mapping")

        timestamp = row["timestamp"]
        if not isinstance(timestamp, (datetime, pd.Timestamp, str)):
            errors.append(f"row {idx}: timestamp has unsupported type")

    if frame["request_id"].astype(str).duplicated().any():
        warnings.append("request_id contains duplicates")
    if (frame["request_id"].astype(str).str.len() == 0).any():
        errors.append("request_id must be non-empty")
    if (frame["policy_name"].astype(str).str.len() == 0).any():
        errors.append("policy_name must be non-empty")

    report = LogValidationReport(len(frame), not errors, errors, warnings)
    if raise_on_error:
        report.raise_for_errors()
    return report
