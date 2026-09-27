"""End-to-end ExploreKit experiment runner owned by participant 4.

Usage:
    python experiments/run_experiment.py --config configs/moderate.yaml
"""
from __future__ import annotations

import argparse
import copy
import inspect
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explorekit.integration import (              
    build_active_policy_for_experiment,
    load_config,
    load_make_replication,
    make_replication,
    project_path,
    public_policy_params,
)
from explorekit.logging import write_logs_parquet              
from explorekit.ope import (              
    DREstimator,
    IPSEstimator,
    SNIPSEstimator,
    compute_reliability,
)
from explorekit.results import ExperimentResult, save_experiment_result              


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _estimate_tuple(result: Any) -> tuple[float | None, float | None, float | None]:
    return (
        _safe_float(result.estimate),
        _safe_float(result.ci_low),
        _safe_float(result.ci_high),
    )


def _expected_summary(
    ips: tuple[float | None, float | None, float | None],
    snips: tuple[float | None, float | None, float | None],
    dr: tuple[float | None, float | None, float | None],
) -> tuple[float | None, str | None, float | None, float | None]:
    for name, values in (("DR", dr), ("SNIPS", snips), ("IPS", ips)):
        if values[0] is not None:
            return values[0], name, values[1], values[2]
    return None, None, None, None


def _can_change_policy(config: dict[str, Any]) -> bool:
    fn = load_make_replication(str(config["simulation"]["provider"]))
    params = inspect.signature(fn).parameters
    return any(name in params for name in ("epsilon", "policy", "evaluation_policy"))


def _compute_ctr_cost(
    config: dict[str, Any],
    policy: Any,
    true_ctr: float,
) -> float | None:
    if not hasattr(policy, "epsilon") or not _can_change_policy(config):
        return None
    epsilon = float(policy.epsilon)
    if np_isclose_zero(epsilon):
        return 0.0

    baseline_config = copy.deepcopy(config)
    baseline_config["policy"].setdefault("overrides", {})["epsilon"] = 0.0
    baseline_config["policy"]["overrides"]["min_epsilon"] = 0.0
    baseline_policy = build_active_policy_for_experiment(baseline_config)
    baseline_replication = make_replication(
        baseline_config,
        baseline_policy,
        seed=int(config["seed"]),
    )
    return float(baseline_replication.true_value - true_ctr)


def np_isclose_zero(value: float) -> bool:
    return abs(value) <= 1e-12


def _conclusion(reliability: str, cold_item_speedup: float | None) -> str:
    if reliability == "Unreliable":
        text = (
            "OPE diagnostics mark this run as Unreliable; the estimate should be "
            "treated as diagnostic until policy overlap or sample size improves."
        )
    elif reliability == "Use with caution":
        text = "OPE is available, but diagnostics require caution when interpreting it."
    else:
        text = "OPE diagnostics did not detect reliability problems for this run."
    if cold_item_speedup is None:
        text += " Cold-item speedup is pending the final participant-3 simulator."
    return text


def run_experiment(
    config: dict[str, Any],
    *,
    save: bool = True,
) -> ExperimentResult:
    """Run simulator -> OPE -> diagnostics -> result persistence."""

    seed = int(config["seed"])
    policy = build_active_policy_for_experiment(config)
    replication = make_replication(config, policy, seed=seed)

    max_weight = config["ope"].get("max_weight")
    bootstrap = config["bootstrap"]
    diagnostics = config["diagnostics"]
    estimators = {
        "ips": IPSEstimator(max_weight=max_weight),
        "snips": SNIPSEstimator(max_weight=max_weight),
        "dr": DREstimator(max_weight=max_weight),
    }

    estimates: dict[str, Any] = {}
    dr_error: str | None = None
    for name, estimator in estimators.items():
        try:
            estimates[name] = estimator.estimate_with_ci(
                replication.ope_input,
                n_bootstrap=int(bootstrap["n_bootstrap"]),
                confidence_level=float(bootstrap["confidence_level"]),
                seed=seed,
            )
        except ValueError as exc:
            if name != "dr":
                raise
            estimates[name] = None
            dr_error = str(exc)

    reliability = compute_reliability(
        replication.ope_input,
        max_weight=max_weight,
        ess_ratio_caution=float(diagnostics["ess_ratio_caution"]),
        ess_ratio_unreliable=float(diagnostics["ess_ratio_unreliable"]),
        clipped_fraction_caution=float(
            diagnostics["clipped_fraction_caution"]
        ),
        support_violation_caution=float(
            diagnostics["support_violation_caution"]
        ),
    )

    ips = _estimate_tuple(estimates["ips"])
    snips = _estimate_tuple(estimates["snips"])
    dr = (
        (None, None, None)
        if estimates["dr"] is None
        else _estimate_tuple(estimates["dr"])
    )
    expected, expected_name, expected_lo, expected_hi = _expected_summary(
        ips, snips, dr
    )

    exploration_share = _safe_float(replication.exploration_share)
    exploration_source = "simulator"
    if exploration_share is None and hasattr(policy, "epsilon"):
        exploration_share = float(policy.epsilon)
        exploration_source = "configured_epsilon"

    ctr_cost = _safe_float(replication.ctr_cost)
    if ctr_cost is None:
        ctr_cost = _compute_ctr_cost(config, policy, replication.true_value)

    cold_item_speedup = _safe_float(replication.cold_item_speedup)
    metadata = dict(replication.metadata)
    metadata.update(
        {
            "dr_error": dr_error,
            "decision_logs_available": replication.logs is not None,
            "exploration_share_source": exploration_source,
            "policy_class": policy.__class__.__name__,
        }
    )

    if save and replication.logs is not None:
        log_paths = write_logs_parquet(
            replication.logs,
            project_path(config["output"]["logs_dir"]),
        )
        metadata["log_paths"] = [str(path) for path in log_paths]

    experiment_cfg = config["experiment"]
    parameters = {
        "policy": public_policy_params(policy),
        "simulation": {
            key: value
            for key, value in config["simulation"].items()
            if key != "provider"
        },
        "cold_start": dict(config["cold_start"]),
        "ope": dict(config["ope"]),
        "bootstrap": dict(config["bootstrap"]),
    }

    result = ExperimentResult(
        experiment_name=str(experiment_cfg["name"]),
        goal=str(experiment_cfg["goal"]),
        preset_name=str(config["preset_name"]),
        logging_policy=str(experiment_cfg.get("logging_policy", "simulator_logging")),
        evaluation_policy=str(replication.ope_input.policy_name),
        n_rounds=len(replication.ope_input),
        seed=seed,
        parameters=parameters,
        true_ctr=float(replication.true_value),
        expected_ctr=expected,
        expected_ctr_estimator=expected_name,
        expected_ci_low=expected_lo,
        expected_ci_high=expected_hi,
        ips_estimate=ips[0],
        ips_ci_low=ips[1],
        ips_ci_high=ips[2],
        snips_estimate=snips[0],
        snips_ci_low=snips[1],
        snips_ci_high=snips[2],
        dr_estimate=dr[0],
        dr_ci_low=dr[1],
        dr_ci_high=dr[2],
        ess=float(reliability.ess),
        ess_ratio=float(reliability.ess_ratio),
        reliability=reliability.status.value,
        reliability_explanation=reliability.explanation,
        exploration_share=exploration_share,
        cold_item_speedup=cold_item_speedup,
        ctr_cost=ctr_cost,
        conclusion=_conclusion(reliability.status.value, cold_item_speedup),
        diagnostics=reliability.to_dict(),
        metadata=metadata,
    )

    if save:
        save_experiment_result(
            result,
            project_path(config["output"]["results_dir"]),
        )
    return result


def _fmt_ctr(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.3f}%"


def _fmt_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "N/A"
    return f"[{low * 100:.3f}%, {high * 100:.3f}%]"


def print_result(result: ExperimentResult) -> None:
    """Print the metrics required by the team definition of done."""

    print("=" * 78)
    print(
        f"ExploreKit | preset={result.preset_name} | "
        f"policy={result.evaluation_policy}"
    )
    print("=" * 78)
    print(f"True CTR:          {_fmt_ctr(result.true_ctr)}")
    print(
        f"Expected CTR:      {_fmt_ctr(result.expected_ctr)} "
        f"({result.expected_ctr_estimator or 'N/A'})"
    )
    expected_ci = _fmt_ci(result.expected_ci_low, result.expected_ci_high)
    print(f"95% CI:            {expected_ci}")
    print(f"IPS:               {_fmt_ctr(result.ips_estimate)}")
    print(f"SNIPS:             {_fmt_ctr(result.snips_estimate)}")
    print(f"DR:                {_fmt_ctr(result.dr_estimate)}")
    print(f"ESS:               {result.ess:.1f} ({result.ess_ratio:.1%})")
    print(f"Reliability:       {result.reliability}")
    print(f"Exploration share: {_fmt_ctr(result.exploration_share)}")
    speedup = (
        "N/A" if result.cold_item_speedup is None else f"{result.cold_item_speedup:.3f}x"
    )
    print(f"Cold-item speedup: {speedup}")
    print(f"CTR cost:          {_fmt_ctr(result.ctr_cost)}")
    print(f"Conclusion:        {result.conclusion}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ExploreKit experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/moderate.yaml"))
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_experiment(config, save=not args.no_save)
    print_result(result)


if __name__ == "__main__":
    main()
