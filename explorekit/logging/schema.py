"""Shared logging contracts for ExploreKit.

The canonical ``Decision`` already belongs to participant 1. Participant 4
reuses that class instead of introducing a second incompatible decision type.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from explorekit.policies import Decision

__all__ = ["Decision", "DecisionLog"]


@dataclass
class DecisionLog:
    """One historical recommendation interaction.

    ``chosen_item`` and ``candidate_items`` are integer action indices from the
    shared action space ``[0, n_actions)``. External string item ids must be
    mapped to these indices before a policy decision is made.
    """

    timestamp: datetime
    request_id: str
    user_segment: str
    context: Sequence[float]
    candidate_items: Sequence[int]
    base_scores: Sequence[float]
    chosen_item: int
    propensity: float
    probabilities: Sequence[float]
    policy_name: str
    policy_params: Mapping[str, Any] = field(default_factory=dict)
    reward: int = 0
    is_exploration: bool = False
    is_cold_item: bool = False

    @classmethod
    def from_decision(
        cls,
        *,
        decision: Decision,
        request_id: str,
        user_segment: str,
        context: Sequence[float],
        candidate_items: Sequence[int],
        base_scores: Sequence[float],
        policy_params: Mapping[str, Any] | None = None,
        reward: int,
        is_cold_item: bool = False,
        timestamp: datetime | None = None,
    ) -> "DecisionLog":
        """Build a log row from participant-1 ``Decision``."""

        return cls(
            timestamp=timestamp or datetime.now(timezone.utc),
            request_id=str(request_id),
            user_segment=str(user_segment),
            context=np.asarray(context, dtype=float).tolist(),
            candidate_items=np.asarray(candidate_items, dtype=int).tolist(),
            base_scores=np.asarray(base_scores, dtype=float).tolist(),
            chosen_item=int(decision.chosen_item),
            propensity=float(decision.propensity),
            probabilities=np.asarray(decision.probabilities, dtype=float).tolist(),
            policy_name=str(decision.policy_name),
            policy_params=dict(policy_params or {}),
            reward=int(reward),
            is_exploration=bool(decision.is_exploration),
            is_cold_item=bool(is_cold_item),
        )

    def to_record(self) -> dict[str, Any]:
        """Convert the dataclass to a DataFrame-friendly dictionary."""

        return asdict(self)
