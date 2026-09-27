"""Сценарий Cold-Item Booster: true CTR, цена exploration и скорость раскрутки.

Запуск:
    python experiments/cold_item_experiment.py
    python experiments/cold_item_experiment.py --n-rounds 800

Epsilon берётся из YAML-пресетов, не из констант в коде.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explorekit.integration import (
    build_active_policy_for_experiment,
    load_config,
    make_replication,
)

PRESET_CONFIGS = (
    "configs/baseline.yaml",
    "configs/conservative.yaml",
    "configs/moderate.yaml",
    "configs/aggressive.yaml",
)


def collect_preset_metrics(
    config_path: str | Path,
    *,
    n_rounds: int | None = None,
    with_ope: bool = False,
    n_bootstrap: int = 200,
) -> dict[str, object]:
    """Один пресет: симулятор + метрики cold-start."""

    config = load_config(config_path)
    if n_rounds is not None:
        config["simulation"]["n_rounds"] = int(n_rounds)
    policy = build_active_policy_for_experiment(config)
    replication = make_replication(config, policy)
    metadata = replication.metadata
    epsilon = float(getattr(policy, "epsilon", float("nan")))
    row: dict[str, object] = {
        "preset": str(config["preset_name"]),
        "epsilon": epsilon,
        "seed": int(config["seed"]),
        "n_rounds": len(replication.ope_input),
        "true_ctr": float(replication.true_value),
        "served_ctr": metadata.get("served_ctr"),
        "ctr_cost": replication.ctr_cost,
        "cold_item_impressions": metadata.get("cold_item_impressions"),
        "time_to_n": metadata.get("mean_time_to_n_eval"),
        "reached_n_eval": metadata.get("reached_n_eval"),
        "speedup": replication.cold_item_speedup,
        "exploration_share": replication.exploration_share,
    }
    if with_ope:
        from explorekit.ope import (
            DREstimator,
            IPSEstimator,
            SNIPSEstimator,
            compute_reliability,
        )

        max_weight = config["ope"].get("max_weight")
        seed = int(config["seed"])
        estimators = {
            "ips": IPSEstimator(max_weight=max_weight),
            "snips": SNIPSEstimator(max_weight=max_weight),
            "dr": DREstimator(max_weight=max_weight),
        }
        for name, estimator in estimators.items():
            estimated = estimator.estimate_with_ci(
                replication.ope_input,
                n_bootstrap=int(n_bootstrap),
                confidence_level=float(config["bootstrap"]["confidence_level"]),
                seed=seed,
            )
            row[f"{name}_estimate"] = estimated.estimate
            row[f"{name}_ci_low"] = estimated.ci_low
            row[f"{name}_ci_high"] = estimated.ci_high
        reliability = compute_reliability(replication.ope_input, max_weight=max_weight)
        row["ess"] = float(reliability.ess)
        row["ess_ratio"] = float(reliability.ess_ratio)
        row["reliability"] = reliability.status.value
    return row


def run_cold_item_experiment(
    *, n_rounds: int | None = None, with_ope: bool = False, n_bootstrap: int = 200
) -> pd.DataFrame:
    """Сводная таблица по четырём exploration-пресетам."""

    rows = [
        collect_preset_metrics(
            path, n_rounds=n_rounds, with_ope=with_ope, n_bootstrap=n_bootstrap
        )
        for path in PRESET_CONFIGS
    ]
    return pd.DataFrame(rows)


def _fmt_ctr(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _fmt_pp(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{float(value) * 100:.2f} pp"


def print_table(frame: pd.DataFrame) -> None:
    header = (
        f"{'preset':<14}{'eps':>6}{'true CTR':>12}{'CTR cost':>12}"
        f"{'cold imps':>12}{'time-to-N':>12}{'speedup':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in frame.itertuples():
        speedup = "N/A" if row.speedup is None or pd.isna(row.speedup) else f"{row.speedup:.3f}x"
        time_to = "N/A" if row.time_to_n is None or pd.isna(row.time_to_n) else f"{row.time_to_n:.1f}"
        cold_imps = (
            "N/A"
            if row.cold_item_impressions is None or pd.isna(row.cold_item_impressions)
            else str(int(row.cold_item_impressions))
        )
        print(
            f"{row.preset:<14}{row.epsilon:>6.2f}{_fmt_ctr(row.true_ctr):>12}"
            f"{_fmt_pp(row.ctr_cost):>12}{cold_imps:>12}{time_to:>12}{speedup:>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-Item Booster experiment")
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=None,
        help="Override simulation.n_rounds from YAML",
    )
    parser.add_argument("--with-ope", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_cold_item_experiment(
        n_rounds=args.n_rounds,
        with_ope=args.with_ope,
        n_bootstrap=args.n_bootstrap,
    )
    print("=" * 78)
    print("Cold-Item Booster: exploration vs exploitation")
    print("=" * 78)
    print_table(frame)
    print()
    print("CTR хранится в [0, 1]; в таблице показан в процентах.")
    print("CTR cost — разница с baseline в процентных пунктах.")


if __name__ == "__main__":
    main()
