"""Эксперимент: OPE на реальном Open Bandit Dataset (задача 8).

Запуск:
    python experiments/open_bandit.py --config configs/ope.yaml

Что делает:
  1. Грузит OBD и приводит к общему формату проекта.
  2. Обучает reward model на одной половине логов (вторая остаётся чистой
     для оценки — иначе DR подглядит в свои же обучающие данные).
  3. Строит три политики exploration поверх base scores.
  4. Считает IPS / SNIPS / DR с bootstrap CI и диагностикой надёжности.
  5. Сверяет IPS/SNIPS/DR с эталонной реализацией Open Bandit Pipeline.

ВРЕМЕННО: epsilon-greedy распределение считается здесь локально
(``_epsilon_greedy_dist``). Когда участник 1 сдаст ``BasePolicy``, эта
функция заменяется на ``policy.action_distribution(context, base_scores)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explorekit.datasets.open_bandit import load_open_bandit_dataset              
from explorekit.ope import (              
    DREstimator,
    IPSEstimator,
    RewardModel,
    SNIPSEstimator,
    build_ope_input,
    compute_reliability,
    split_for_reward_model,
)


def _epsilon_greedy_dist(base_scores: np.ndarray, epsilon: float) -> np.ndarray:
    """P(best) = 1 - eps + eps/N, P(other) = eps/N. Временная замена участника 1."""
    n, n_actions = base_scores.shape
    best = np.argmax(base_scores, axis=1)
    dist = np.full((n, n_actions), epsilon / n_actions)
    dist[np.arange(n), best] += 1.0 - epsilon
    return dist


def _load_or_exit(obd_config: dict):
    """Грузит датасет, а при его отсутствии выходит с понятной подсказкой.

    Данные намеренно не лежат в Git (правило участника 4), поэтому в свежем
    клоне репозитория это штатная ситуация, а не ошибка кода.
    """
    try:
        return load_open_bandit_dataset(
            data_dir=obd_config["data_dir"],
            campaign=obd_config["campaign"],
            behavior_policy=obd_config["behavior_policy"],
        )
    except FileNotFoundError as error:
        print(f"Датасет не найден.\n  {error}")
        print("Подсказка: bash scripts/download_obd.sh")
        raise SystemExit(1) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPE на Open Bandit Dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/ope.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = config["seed"]
    max_weight = config["ope"]["max_weight"]
    obd_config = config["open_bandit"]
    bootstrap_config = config["bootstrap"]
    rm_config = config["reward_model"]
    diag_config = config["diagnostics"]

    print("=" * 84)
    print("ШАГ 1. Загрузка Open Bandit Dataset")
    print("=" * 84)
    dataset = _load_or_exit(obd_config)
    n = len(dataset)
    print(
        f"Раундов: {n}, действий: {dataset.n_actions}, "
        f"контекст: {dataset.context.shape[1]} признаков"
    )
    print(
        f"CTR логирующей политики ({obd_config['behavior_policy']}): "
        f"{dataset.logs['reward'].mean():.4%}"
    )
    print(f"Кликов в логах: {int(dataset.logs['reward'].sum())}")

    print("\n" + "=" * 84)
    print("ШАГ 2. Reward model (обучение на отдельной половине логов)")
    print("=" * 84)
    train_idx, eval_idx = split_for_reward_model(
        n, train_fraction=rm_config["train_fraction"], seed=seed
    )
    chosen = dataset.logs["chosen_item"].to_numpy(dtype=int)
    chosen_features = dataset.action_features[np.arange(n), chosen]

    reward_model = RewardModel(
        C=rm_config["C"], max_iter=rm_config["max_iter"], seed=seed
    ).fit(
        context=dataset.context[train_idx],
        action_features_chosen=chosen_features[train_idx],
        reward=dataset.logs["reward"].to_numpy(dtype=float)[train_idx],
        n_actions=dataset.n_actions,
    )
    print(f"Обучено на {len(train_idx)} раундах, оценка пойдёт по {len(eval_idx)}")

    eval_logs = dataset.logs.iloc[eval_idx].reset_index(drop=True)
    eval_context = dataset.context[eval_idx]
    eval_action_features = dataset.action_features[eval_idx]
    q_hat_all = reward_model.predict_all_actions(eval_context, eval_action_features)
    base_scores = q_hat_all                                   
    top1_ctr = base_scores.max(axis=1).mean()
    print(f"Средний предсказанный CTR топ-1 действия: {top1_ctr:.4%}")

    print("\n" + "=" * 84)
    print("ШАГ 3. Офлайн-оценка политик exploration")
    print("=" * 84)
    estimators = [
        IPSEstimator(max_weight=max_weight),
        SNIPSEstimator(max_weight=max_weight),
        DREstimator(max_weight=max_weight),
    ]

    for policy_name, epsilon in obd_config["evaluation_epsilons"].items():
        action_dist = _epsilon_greedy_dist(base_scores, epsilon)
        ope_input = build_ope_input(
            logs=eval_logs,
            evaluation_action_dist=action_dist,
            policy_name=f"{policy_name}(eps={epsilon})",
            q_hat_all_actions=q_hat_all,
        )

        print(f"\nПолитика: {policy_name} (epsilon={epsilon})")
        header = f"{'  Оценщик':<20}{'CTR':>10}{'95% CI':>26}"
        print(header)
        print("-" * len(header))
        for estimator in estimators:
            result = estimator.estimate_with_ci(
                ope_input,
                n_bootstrap=bootstrap_config["n_bootstrap"],
                confidence_level=bootstrap_config["confidence_level"],
                seed=seed,
            )
            ci = f"[{result.ci_low * 100:.3f}%, {result.ci_high * 100:.3f}%]"
            print(f"  {result.estimator_name:<18}{result.estimate * 100:>9.3f}%{ci:>26}")

        reliability = compute_reliability(
            ope_input,
            max_weight=max_weight,
            ess_ratio_caution=diag_config["ess_ratio_caution"],
            ess_ratio_unreliable=diag_config["ess_ratio_unreliable"],
            clipped_fraction_caution=diag_config["clipped_fraction_caution"],
            support_violation_caution=diag_config["support_violation_caution"],
        )
        print(f"  Статус: {reliability.status.value}")
        print(
            f"  ESS = {reliability.ess:.0f}/{reliability.n_rounds} "
            f"({reliability.ess_ratio:.1%}), max weight = {reliability.max_weight:.1f}, "
            f"обрезано {reliability.fraction_clipped:.1%}"
        )
        print(
            f"  Дисперсия весов: var = {reliability.weight_variance:.2f}, "
            f"CV = {reliability.weight_cv:.1f}, p99 = {reliability.weight_p99:.1f}, "
            f"se(IPS) = {reliability.ips_standard_error * 100:.4f} п.п."
        )
        print(f"  {reliability.explanation}")

    print("\n" + "=" * 84)
    print("ШАГ 4. Сверка с эталонной реализацией Open Bandit Pipeline")
    print("=" * 84)
    try:
        from experiments._obp_crosscheck import (
            obp_available,
            reference_dr,
            reference_ips,
            reference_snips,
        )
    except ImportError:
        obp_available = False

    if not obp_available:
        print("obp недоступен в этом окружении — сверка пропущена.")
        return

    epsilon = obd_config["evaluation_epsilons"]["moderate"]
    action_dist = _epsilon_greedy_dist(base_scores, epsilon)
    ope_input = build_ope_input(
        logs=eval_logs,
        evaluation_action_dist=action_dist,
        policy_name=f"moderate(eps={epsilon})",
        q_hat_all_actions=q_hat_all,
    )
    eval_action = eval_logs["chosen_item"].to_numpy(dtype=int)
    eval_pscore = eval_logs["propensity"].to_numpy(dtype=float)
    eval_reward = eval_logs["reward"].to_numpy(dtype=float)

    comparisons = [
        (
            "IPS",
            IPSEstimator().estimate(ope_input),
            reference_ips(eval_reward, eval_action, eval_pscore, action_dist),
        ),
        (
            "SNIPS",
            SNIPSEstimator().estimate(ope_input),
            reference_snips(eval_reward, eval_action, eval_pscore, action_dist),
        ),
        (
            "DR",
            DREstimator().estimate(ope_input),
            reference_dr(eval_reward, eval_action, eval_pscore, action_dist, q_hat_all),
        ),
    ]
    header = f"{'  Оценщик':<12}{'наш':>14}{'obp':>14}{'расхождение':>16}"
    print(header)
    print("-" * len(header))
    for name, ours, reference in comparisons:
        gap = abs(ours - reference)
        print(f"  {name:<10}{ours:>14.10f}{reference:>14.10f}{gap:>16.2e}")
    print(
        "\n(сверка без клиппинга — в obp клиппинг задаётся иначе, "
        "сравниваем базовые формулы)"
    )


if __name__ == "__main__":
    main()
