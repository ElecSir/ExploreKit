"""Загрузка Open Bandit Dataset и приведение к общему формату проекта (задача 8).

Датасет: Saito, Aihara, Matsutani & Narita, "Open Bandit Dataset and Pipeline:
Towards Realistic and Reproducible Off-Policy Evaluation", NeurIPS 2021
Datasets & Benchmarks Track (https://arxiv.org/abs/2008.07146).
Репозиторий: https://github.com/st-tech/zr-obp

Почему именно ``behavior_policy="random"``: в этой части датасета логирующая
политика — честный uniform random, то есть propensity записаны точно
(1/80 на показ), а не восстановлены постфактум. Для проверки корректности
наших оценщиков это критично: любое расхождение с реальностью тогда
списывается на оценщик, а не на кривые propensity.

По правилу участника 4 (задача 4) сам датасет в Git не кладётся — здесь
только загрузчик; путь к данным передаётся аргументом или берётся из
``data/obd/``.

Формат на выходе — ``pandas.DataFrame`` в схеме DecisionLog (участник 4),
плюс отдельно массив признаков действий, нужный reward model для DR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["OpenBanditData", "load_open_bandit_dataset"]

N_ACTIONS = 80
AFFINITY_PREFIX = "user-item_affinity_"
USER_FEATURE_PREFIX = "user_feature_"


@dataclass
class OpenBanditData:
    """OBD в формате проекта.

    Attributes:
        logs: DataFrame в схеме DecisionLog — колонки ``timestamp``,
            ``request_id``, ``chosen_item``, ``propensity``, ``reward``,
            ``position``, ``policy_name``.
        context: ``(n, d_context)`` признаки раунда (пользователь + позиция).
        action_features: ``(n, n_actions, d_action)`` признаки каждого
            действия-кандидата — здесь это affinity пользователя к item-у.
        n_actions: размер пространства действий.
    """

    logs: pd.DataFrame
    context: np.ndarray
    action_features: np.ndarray
    n_actions: int = N_ACTIONS

    def __len__(self) -> int:
        return len(self.logs)


def load_open_bandit_dataset(
    data_dir: str | Path,
    campaign: str = "all",
    behavior_policy: str = "random",
    max_user_feature_levels: int = 15,
) -> OpenBanditData:
    """Читает OBD и конвертирует в общий формат проекта.

    Args:
        data_dir: корень с данными OBD. Ожидается структура
            ``<data_dir>/<behavior_policy>/<campaign>/all.csv``.
        campaign: ``all`` / ``men`` / ``women``.
        behavior_policy: ``random`` или ``bts``.
        max_user_feature_levels: сколько самых частых уровней оставлять у
            хэшированных категориальных признаков пользователя, остальное
            схлопывается в ``__other__`` (иначе one-hot раздувается на хэшах).

    Returns:
        :class:`OpenBanditData`.

    Raises:
        FileNotFoundError: если CSV не найден по ожидаемому пути.
    """
    root = Path(data_dir) / behavior_policy / campaign
    csv_path = root / f"{campaign}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"не найден {csv_path}. Скачайте Open Bandit Dataset "
            f"(https://github.com/st-tech/zr-obp) и распакуйте в {data_dir}."
        )

    raw = pd.read_csv(csv_path, index_col=0)

    affinity_cols = sorted(
        (c for c in raw.columns if c.startswith(AFFINITY_PREFIX)),
        key=lambda c: int(c.split("_")[-1]),
    )
    if len(affinity_cols) != N_ACTIONS:
        raise ValueError(
            f"ожидалось {N_ACTIONS} affinity-колонок, найдено {len(affinity_cols)}"
        )

                                                                            
    affinity = raw[affinity_cols].to_numpy(dtype=float)                  
    action_features = affinity[:, :, None]                     

                                                                       
    user_cols = [c for c in raw.columns if c.startswith(USER_FEATURE_PREFIX)]
    user_blocks = []
    for col in user_cols:
        top_levels = raw[col].value_counts().nlargest(max_user_feature_levels).index
        collapsed = raw[col].where(raw[col].isin(top_levels), other="__other__")
        user_blocks.append(pd.get_dummies(collapsed, prefix=col))
    position_onehot = pd.get_dummies(raw["position"], prefix="position")
    context = pd.concat(user_blocks + [position_onehot], axis=1).to_numpy(dtype=float)

                                                   
    n = len(raw)
    logs = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["timestamp"], format="mixed", utc=True),
            "request_id": np.arange(n),
            "chosen_item": raw["item_id"].to_numpy(dtype=int),
            "propensity": raw["propensity_score"].to_numpy(dtype=float),
            "reward": raw["click"].to_numpy(dtype=float),
            "position": raw["position"].to_numpy(dtype=int),
            "policy_name": f"obd_{behavior_policy}",
        }
    )

    return OpenBanditData(
        logs=logs,
        context=context,
        action_features=action_features,
        n_actions=N_ACTIONS,
    )
