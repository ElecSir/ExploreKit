from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.run_experiment import run_experiment
from explorekit.integration import (
    ActionIndexMap,
    build_active_policy_for_experiment,
    build_ope_input_from_logs,
    load_config,
    run_smoke,
)
from explorekit.logging import Decision as LoggingDecision
from explorekit.logging import DecisionLog, validate_logs
from explorekit.policies import Decision as PolicyDecision
from explorekit.policies import EpsilonGreedyPolicy
from explorekit.simulator.environment import N_ACTIONS


def test_shared_decision_is_participant1_decision() -> None:
    assert LoggingDecision is PolicyDecision


def test_action_index_map_encodes_string_item_ids() -> None:
    mapping = ActionIndexMap.from_items(["sku-a", "sku-b", "sku-c"])
    assert mapping.n_actions == 3
    assert mapping.encode("sku-b") == 1
    assert mapping.encode_many(["sku-c", "sku-a"]) == [2, 0]
    assert mapping.decode(1) == "sku-b"


def test_decision_log_accepts_candidate_subset_of_action_space() -> None:
    policy = EpsilonGreedyPolicy(n_actions=5, epsilon=0.2, seed=7)
    context = np.array([0.2, -0.1, 1.0])
    base_scores = np.array([0.1, 0.9, -0.2, 0.8, 0.4])
    candidates = np.array([1, 3, 4])
    decision = policy.select_action(context, base_scores, candidates)
    log = DecisionLog.from_decision(
        decision=decision,
        request_id="req-1",
        user_segment="a",
        context=context,
        candidate_items=candidates,
        base_scores=base_scores,
        policy_params={"epsilon": 0.2},
        reward=1,
        timestamp=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )
    report = validate_logs(pd.DataFrame([log.to_record()]), raise_on_error=False)
    assert report.valid, report.errors


def test_validator_rejects_string_chosen_item() -> None:
    row = {
        "timestamp": datetime(2026, 9, 20, tzinfo=timezone.utc),
        "request_id": "req-1",
        "user_segment": "a",
        "context": [1.0, 2.0],
        "candidate_items": [0, 1],
        "base_scores": [0.2, 0.8],
        "chosen_item": "sku-b",
        "propensity": 0.5,
        "probabilities": [0.5, 0.5],
        "policy_name": "test",
        "policy_params": {},
        "reward": 1,
        "is_exploration": False,
        "is_cold_item": False,
    }
    report = validate_logs(pd.DataFrame([row]), raise_on_error=False)
    assert not report.valid
    assert any("chosen_item must be int" in error for error in report.errors)


def test_varying_candidate_sets_policy_to_ope_contract() -> None:
    policy = EpsilonGreedyPolicy(n_actions=4, epsilon=0.3, seed=11)
    rng = np.random.default_rng(11)
    rows = []
    candidate_sets = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    for i in range(20):
        context = rng.normal(size=3)
        base_scores = rng.normal(size=4)
        candidates = candidate_sets[i % 2]
        decision = policy.select_action(context, base_scores, candidates)
        rows.append(
            DecisionLog.from_decision(
                decision=decision,
                request_id=f"r-{i}",
                user_segment="s",
                context=context,
                candidate_items=candidates,
                base_scores=base_scores,
                policy_params={"epsilon": policy.epsilon},
                reward=int(i % 5 == 0),
            ).to_record()
        )
    logs = pd.DataFrame(rows)
    ope_input = build_ope_input_from_logs(logs, policy)
    assert len(ope_input) == len(logs)
    assert np.allclose(ope_input.importance_weights(), 1.0)


def test_real_modules_smoke_1000_logs() -> None:
    result = run_smoke(seed=42, n_rounds=1000)
    assert result.n_logs == 1000
    assert np.isfinite(result.ips_estimate)
    assert result.ips_estimate == pytest.approx(result.mean_reward)
    assert result.ess == pytest.approx(1000.0)


def test_moderate_config_uses_real_policy_factory() -> None:
    config = load_config("configs/moderate.yaml")
    policy = build_active_policy_for_experiment(config)
    assert isinstance(policy, EpsilonGreedyPolicy)
    assert policy.epsilon == pytest.approx(0.10)
    assert policy.n_actions == N_ACTIONS


@pytest.mark.parametrize(
    ("preset", "epsilon"),
    [
        ("baseline", 0.00),
        ("conservative", 0.05),
        ("moderate", 0.10),
        ("aggressive", 0.25),
    ],
)
def test_all_preset_configs(preset: str, epsilon: float) -> None:
    config = load_config(f"configs/{preset}.yaml")
    policy = build_active_policy_for_experiment(config)
    assert policy.epsilon == pytest.approx(epsilon)


def test_end_to_end_runner(tmp_path: Path) -> None:
    config = load_config("configs/moderate.yaml")
    config["simulation"]["n_rounds"] = 300
    config["bootstrap"]["n_bootstrap"] = 20
    config["output"]["results_dir"] = str(tmp_path / "results")
    config["output"]["logs_dir"] = str(tmp_path / "logs")

    result = run_experiment(config, save=True)
    assert result.n_rounds == 300
    assert result.true_ctr is not None and np.isfinite(result.true_ctr)
    assert result.ips_estimate is not None and np.isfinite(result.ips_estimate)
    assert result.snips_estimate is not None and np.isfinite(result.snips_estimate)
    assert result.dr_estimate is not None and np.isfinite(result.dr_estimate)
    assert np.isfinite(result.ess)
    assert result.reliability in {"Reliable", "Use with caution", "Unreliable"}
    assert result.exploration_share is not None
    assert np.isfinite(result.exploration_share)
    assert result.cold_item_speedup is not None
    assert np.isfinite(result.cold_item_speedup)
    assert result.ctr_cost is not None and np.isfinite(result.ctr_cost)
    assert list((tmp_path / "results").glob("*.json"))
    assert list((tmp_path / "logs").glob("date=*/*.parquet"))


def test_adapter_forwards_cold_start_kwargs() -> None:
    config = load_config("configs/moderate.yaml")
    policy = build_active_policy_for_experiment(config)
    from explorekit.integration.simulator_adapter import _supported_kwargs
    from explorekit.simulator.environment import make_replication as sim_fn

    kwargs = _supported_kwargs(sim_fn, config, policy)
    assert kwargs["max_age_days"] == 7
    assert kwargs["min_impressions"] == 20
    assert kwargs["boost_multiplier"] == pytest.approx(2.0)
    assert kwargs["time_to_n_threshold"] == 20
    assert kwargs["logging_epsilon"] == pytest.approx(0.5)
    assert kwargs["cold_penalty"] == pytest.approx(1.2)
    assert "policy" in kwargs


def test_runner_is_reproducible_for_same_seed() -> None:
    config = load_config("configs/conservative.yaml")
    config["simulation"]["n_rounds"] = 250
    config["bootstrap"]["n_bootstrap"] = 20
    first = run_experiment(config, save=False)
    second = run_experiment(config, save=False)
    assert first.true_ctr == pytest.approx(second.true_ctr)
    assert first.ips_estimate == pytest.approx(second.ips_estimate)
    assert first.snips_estimate == pytest.approx(second.snips_estimate)
    assert first.dr_estimate == pytest.approx(second.dr_estimate)
