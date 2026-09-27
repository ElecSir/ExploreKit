"""Эксперимент: валидация оценщиков на синтетике с известной истиной (задача 7).

Запуск:
    python experiments/ope_synthetic.py --config configs/ope.yaml

Что выводит:
  1. Таблицу одного прогона: True CTR vs IPS/SNIPS/DR, абсолютная ошибка, 95% CI.
  2. Агрегат по 50 прогонам: Bias, Std, RMSE, CI coverage.

Источник данных: ``explorekit.simulator.environment.make_replication``.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explorekit.simulator.environment import make_replication              
from explorekit.ope import (              
    DREstimator,
    IPSEstimator,
    SNIPSEstimator,
    compute_reliability,
)
from explorekit.ope.validation import (              
    run_single_comparison,
    run_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic validation OPE-оценщиков")
    parser.add_argument("--config", type=Path, default=Path("configs/ope.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = config["seed"]
    max_weight = config["ope"]["max_weight"]
    sv_config = config["synthetic_validation"]
    confidence_level = config["bootstrap"]["confidence_level"]

    estimators = [
        IPSEstimator(max_weight=max_weight),
        SNIPSEstimator(max_weight=max_weight),
        DREstimator(max_weight=max_weight),
    ]

    replication_fn = partial(
        make_replication,
        n_rounds=sv_config["n_rounds"],
        epsilon=sv_config["epsilon"],
        compute_cold_metrics=False,
    )

    print("=" * 78)
    print("ЧАСТЬ 1. Один прогон: True CTR против оценок")
    print("=" * 78)
    single = replication_fn(seed)
    table = run_single_comparison(
        single,
        estimators,
        n_bootstrap=config["bootstrap"]["n_bootstrap"],
        confidence_level=confidence_level,
        seed=seed,
    )
    print(
        f"Истинный CTR политики: {single.true_value:.4f} "
        f"({single.true_value * 100:.2f}%)\n"
    )
    header = (
        f"{'Оценщик':<18}{'Оценка':>10}{'|ошибка|':>11}" f"{'95% CI':>24}{'покрыл':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in table.itertuples():
        ci = f"[{row.ci_low * 100:.2f}%, {row.ci_high * 100:.2f}%]"
        mark = "да" if row.covers_true else "НЕТ"
        print(
            f"{row.estimator:<18}{row.estimate:>10.4f}{row.abs_error:>11.4f}"
            f"{ci:>24}{mark:>9}"
        )

    reliability = compute_reliability(single.ope_input, max_weight=max_weight)
    print(f"\nДиагностика: {reliability.status.value}")
    print(
        f"  ESS = {reliability.ess:.0f} из {reliability.n_rounds} "
        f"(ESS/N = {reliability.ess_ratio:.1%}), "
        f"max weight = {reliability.max_weight:.1f}, "
        f"обрезано {reliability.fraction_clipped:.1%}"
    )
    print(f"  {reliability.explanation}")

    print("\n" + "=" * 78)
    n_reps = sv_config["n_replications"]
    print(f"ЧАСТЬ 2. {n_reps} прогонов: Bias, Std, RMSE, CI coverage")
    print("=" * 78)
    report = run_validation(
        replication_fn,
        estimators,
        n_replications=sv_config["n_replications"],
        n_bootstrap=sv_config["n_bootstrap"],
        confidence_level=confidence_level,
        seed=seed,
    )
    print(report.format_table())
    print(
        f"\nCI coverage должен быть близок к заявленным "
        f"{confidence_level:.0%} — это проверка честности доверительных интервалов."
    )


if __name__ == "__main__":
    main()
