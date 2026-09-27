"""Сверка наших оценщиков с эталонной реализацией Open Bandit Pipeline (задача 8).

Зачем: наши IPS/SNIPS/DR написаны с нуля, и одних unit-тестов мало —
они проверяют формулы против моих же ожиданий. Сверка с независимой
реализацией авторов датасета ловит систематические ошибки, которые
собственные тесты пропустили бы.

Техническая оговорка: библиотека ``obp`` импортирует ``torch`` на верхнем
уровне (``obp/utils.py``), хотя использует его ТОЛЬКО для аннотаций типов и
для отдельной тензорной ветки ``estimate_policy_value_tensor``. Numpy-путь,
который нам и нужен, к torch не обращается. Ставить ~8 ГБ CUDA-зависимостей
ради аннотаций бессмысленно, поэтому здесь подставляется минимальная
заглушка torch. Если в вашем окружении torch уже стоит, заглушка не
активируется и используется настоящий пакет.
"""

from __future__ import annotations

import sys
import types

import numpy as np

__all__ = ["obp_available", "reference_ips", "reference_snips", "reference_dr"]


def _install_torch_shim() -> None:
    """Ставит заглушку torch, если настоящего нет.

    Покрывает только то, к чему ``obp`` обращается НА ИМПОРТЕ (аннотации,
    словари слоёв в ``obp.policy.offline``). Любая попытка реально обучить
    нейросетевую политику через эту заглушку упадёт — и правильно сделает,
    нам нужны исключительно numpy-оценщики из ``obp.ope``.
    """
    try:
        import torch              

        return
    except ImportError:
        pass

    torch_mod = types.ModuleType("torch")

    class _Tensor:                                              
        pass

    class _Stub:
        """Заглушка слоя/оптимизатора: конструируется, но не работает."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "torch-заглушка: нейросетевые компоненты obp недоступны. "
                "Для них установите настоящий torch."
            )

    torch_mod.Tensor = _Tensor
    torch_mod.int64 = np.int64
    torch_mod.float32 = np.float32
    torch_mod.ones_like = np.ones_like
    torch_mod.allclose = np.allclose
    torch_mod.any = np.any
    torch_mod.all = np.all
    torch_mod.from_numpy = lambda x: x
    torch_mod.manual_seed = lambda seed: None

    nn_mod = types.ModuleType("torch.nn")
    for layer in (
        "Identity",
        "Linear",
        "ReLU",
        "Sequential",
        "Sigmoid",
        "Softmax",
        "Tanh",
        "ELU",
        "LeakyReLU",
        "Module",
        "Parameter",
    ):
        setattr(nn_mod, layer, type(layer, (_Stub,), {}))
    nn_functional = types.ModuleType("torch.nn.functional")
    nn_mod.functional = nn_functional

    optim_mod = types.ModuleType("torch.optim")
    for opt in ("Adam", "LBFGS", "SGD", "AdamW", "Optimizer"):
        setattr(optim_mod, opt, type(opt, (_Stub,), {}))

    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    for cls in ("DataLoader", "Dataset", "TensorDataset"):
        setattr(data_mod, cls, type(cls, (_Stub,), {}))
    utils_mod.data = data_mod

    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_mod.utils = utils_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_functional
    sys.modules["torch.optim"] = optim_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod


def _try_import_obp():
    _install_torch_shim()
    try:
        from obp.ope import (
            DoublyRobust,
            InverseProbabilityWeighting,
            SelfNormalizedInverseProbabilityWeighting,
        )

        return (
            InverseProbabilityWeighting,
            SelfNormalizedInverseProbabilityWeighting,
            DoublyRobust,
        )
    except Exception:
        return None


_OBP = _try_import_obp()
obp_available = _OBP is not None


def _to_obp_format(
    action: np.ndarray,
    evaluation_action_dist: np.ndarray,
) -> np.ndarray:
    """obp ждёт action_dist формы (n_rounds, n_actions, len_list)."""
    return evaluation_action_dist[:, :, None]


def reference_ips(
    reward: np.ndarray,
    action: np.ndarray,
    pscore: np.ndarray,
    evaluation_action_dist: np.ndarray,
) -> float:
    """Эталонный IPS из obp."""
    if not obp_available:
        raise RuntimeError("obp недоступен")
    estimator = _OBP[0]()
    return float(
        estimator.estimate_policy_value(
            reward=reward,
            action=action,
            pscore=pscore,
            action_dist=_to_obp_format(action, evaluation_action_dist),
            position=np.zeros(len(reward), dtype=int),
        )
    )


def reference_snips(
    reward: np.ndarray,
    action: np.ndarray,
    pscore: np.ndarray,
    evaluation_action_dist: np.ndarray,
) -> float:
    """Эталонный SNIPS из obp."""
    if not obp_available:
        raise RuntimeError("obp недоступен")
    estimator = _OBP[1]()
    return float(
        estimator.estimate_policy_value(
            reward=reward,
            action=action,
            pscore=pscore,
            action_dist=_to_obp_format(action, evaluation_action_dist),
            position=np.zeros(len(reward), dtype=int),
        )
    )


def reference_dr(
    reward: np.ndarray,
    action: np.ndarray,
    pscore: np.ndarray,
    evaluation_action_dist: np.ndarray,
    q_hat_all_actions: np.ndarray,
) -> float:
    """Эталонный Doubly Robust из obp."""
    if not obp_available:
        raise RuntimeError("obp недоступен")
    estimator = _OBP[2]()
    return float(
        estimator.estimate_policy_value(
            reward=reward,
            action=action,
            pscore=pscore,
            action_dist=_to_obp_format(action, evaluation_action_dist),
            estimated_rewards_by_reg_model=q_hat_all_actions[:, :, None],
            position=np.zeros(len(reward), dtype=int),
        )
    )
