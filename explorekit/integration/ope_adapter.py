"""Boundary between DecisionLog, participant-1 policies and participant-2 OPE."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from explorekit.logging import validate_logs
from explorekit.ope import OPEInput, build_ope_input

__all__ = [
    "build_ope_input_from_logs",
    "evaluation_distribution",
    "stack_policy_inputs",
]


def _stack_numeric(values: pd.Series, name: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    width: int | None = None
    for idx, value in values.items():
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {idx}: {name} is not numeric") from exc
        if arr.ndim != 1:
            raise ValueError(f"row {idx}: {name} must be a 1D vector")
        if width is None:
            width = len(arr)
        elif len(arr) != width:
            raise ValueError(f"row {idx}: inconsistent {name} width")
        rows.append(arr)
    if not rows:
        raise ValueError("DecisionLog is empty")
    return np.vstack(rows)


def stack_policy_inputs(logs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract context and full action-score matrices from DecisionLog."""

    context = _stack_numeric(logs["context"], "context")
    base_scores = _stack_numeric(logs["base_scores"], "base_scores")
    return context, base_scores


def _candidate_rows(logs: pd.DataFrame) -> list[np.ndarray]:
    return [np.asarray(value, dtype=int) for value in logs["candidate_items"]]


def evaluation_distribution(
    policy: Any,
    context: np.ndarray,
    base_scores: np.ndarray,
    candidate_rows: list[np.ndarray],
) -> np.ndarray:
    """Evaluate a policy on a log batch, respecting candidate subsets per row."""

    if not hasattr(policy, "action_distribution"):
        raise TypeError("policy must expose action_distribution")
    if len(candidate_rows) != len(context):
        raise ValueError("candidate_rows and context lengths differ")

    same_candidates = all(
        np.array_equal(candidate_rows[0], candidates) for candidates in candidate_rows[1:]
    )
    if same_candidates:
        dist = np.asarray(
            policy.action_distribution(
                context,
                base_scores,
                candidate_items=candidate_rows[0],
            ),
            dtype=float,
        )
    else:
        rows = []
        for i, candidates in enumerate(candidate_rows):
            row = policy.action_distribution(
                context[i],
                base_scores[i],
                candidate_items=candidates,
            )
            rows.append(np.asarray(row, dtype=float))
        dist = np.vstack(rows)

    expected = (len(context), base_scores.shape[1])
    if dist.shape != expected:
        raise ValueError(
            f"action_distribution returned {dist.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(dist)) or np.any(dist < 0.0):
        raise ValueError("action_distribution contains invalid probabilities")
    if not np.allclose(dist.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("action_distribution rows must sum to 1")
    return dist


def build_ope_input_from_logs(
    logs: pd.DataFrame,
    policy: Any,
    *,
    policy_name: str | None = None,
    q_hat_all_actions: np.ndarray | None = None,
) -> OPEInput:
    """Convert validated DecisionLog plus evaluation policy to ``OPEInput``."""

    validate_logs(logs)
    context, base_scores = stack_policy_inputs(logs)
    candidate_rows = _candidate_rows(logs)
    dist = evaluation_distribution(policy, context, base_scores, candidate_rows)
    name = policy_name or getattr(policy, "name", policy.__class__.__name__)
    return build_ope_input(
        logs=logs,
        evaluation_action_dist=dist,
        policy_name=str(name),
        q_hat_all_actions=q_hat_all_actions,
    )
