"""Small real-module integration smoke used before the final simulator is ready."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from explorekit.integration.config import project_path
from explorekit.integration.ope_adapter import build_ope_input_from_logs
from explorekit.logging import DecisionLog, validate_logs
from explorekit.ope import IPSEstimator, compute_reliability
from explorekit.policies import build_active_policy, load_policies_config

__all__ = ["SmokeResult", "run_smoke"]


@dataclass(frozen=True)
class SmokeResult:
    """Summary of the participant 1 -> 4 -> 2 integration smoke."""

    n_logs: int
    mean_reward: float
    ips_estimate: float
    ess: float
    reliability: str
    logs: pd.DataFrame


def run_smoke(seed: int = 42, n_rounds: int = 1000) -> SmokeResult:
    """Run epsilon-greedy -> DecisionLog -> OPE on 1000 synthetic interactions."""

    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    cfg = deepcopy(load_policies_config(project_path("configs/policies.yaml")))
    cfg["active_policy"] = "smoke_epsilon_greedy"
    cfg["seed"] = seed
    cfg["policies"]["smoke_epsilon_greedy"]["seed"] = seed
    policy = build_active_policy(cfg)

    rng = np.random.default_rng(seed)
    candidates = np.arange(policy.n_actions, dtype=int)
    rows: list[dict] = []
    start = datetime(2026, 9, 20, tzinfo=timezone.utc)

    for i in range(n_rounds):
        context = rng.normal(size=3)
        base_scores = rng.normal(size=policy.n_actions)
        decision = policy.select_action(context, base_scores, candidates)
        click_prob = 1.0 / (1.0 + np.exp(3.0 - 0.5 * base_scores[decision.chosen_item]))
        reward = int(rng.binomial(1, click_prob))
        log = DecisionLog.from_decision(
            decision=decision,
            request_id=f"smoke-{i}",
            user_segment="smoke",
            context=context,
            candidate_items=candidates,
            base_scores=base_scores,
            policy_params={"epsilon": float(policy.epsilon)},
            reward=reward,
            is_cold_item=False,
            timestamp=start + timedelta(seconds=i),
        )
        rows.append(log.to_record())

    logs = pd.DataFrame(rows)
    validate_logs(logs)
    ope_input = build_ope_input_from_logs(logs, policy)
    ips = IPSEstimator(max_weight=None).estimate(ope_input)
    reliability = compute_reliability(ope_input, max_weight=None)
    return SmokeResult(
        n_logs=len(logs),
        mean_reward=float(logs["reward"].mean()),
        ips_estimate=float(ips),
        ess=float(reliability.ess),
        reliability=reliability.status.value,
        logs=logs,
    )
