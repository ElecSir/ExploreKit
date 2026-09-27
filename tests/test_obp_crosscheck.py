"""Сверка с эталонной реализацией Open Bandit Pipeline (задача 8).

Unit-тесты проверяют наши формулы против моих же ожиданий. Этот тест
проверяет их против независимой реализации авторов датасета — он ловит
systematic errors, которые собственные тесты пропустили бы по построению.

Если ``obp`` в окружении недоступен, тесты пропускаются (skip), а не падают:
сверка полезна, но не должна блокировать CI всей команды.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments._obp_crosscheck import (
    obp_available,
    reference_dr,
    reference_ips,
    reference_snips,
)
from explorekit.ope import DREstimator, IPSEstimator, OPEInput, SNIPSEstimator

pytestmark = pytest.mark.skipif(
    not obp_available, reason="obp недоступен в этом окружении"
)


@pytest.fixture
def scenario():
    rng = np.random.default_rng(0)
    n, n_actions = 3000, 6
    action = rng.integers(0, n_actions, size=n)
    pscore = np.full(n, 1.0 / n_actions)
    reward = rng.binomial(1, 0.15, size=n).astype(float)
    dist = rng.dirichlet(np.ones(n_actions), size=n)
    q_hat = rng.uniform(0.05, 0.3, size=(n, n_actions))

    rows = np.arange(n)
    data = OPEInput(
        reward=reward,
        behavior_pscore=pscore,
        evaluation_pscore=dist[rows, action],
        q_hat_logged=q_hat[rows, action],
        q_hat_policy=np.sum(dist * q_hat, axis=1),
    )
    return {
        "data": data,
        "action": action,
        "pscore": pscore,
        "reward": reward,
        "dist": dist,
        "q_hat": q_hat,
    }


def test_ips_matches_reference(scenario) -> None:
    ours = IPSEstimator().estimate(scenario["data"])
    reference = reference_ips(
        scenario["reward"], scenario["action"], scenario["pscore"], scenario["dist"]
    )
    assert ours == pytest.approx(reference, rel=1e-12)


def test_snips_matches_reference(scenario) -> None:
    ours = SNIPSEstimator().estimate(scenario["data"])
    reference = reference_snips(
        scenario["reward"], scenario["action"], scenario["pscore"], scenario["dist"]
    )
    assert ours == pytest.approx(reference, rel=1e-12)


def test_dr_matches_reference(scenario) -> None:
    ours = DREstimator().estimate(scenario["data"])
    reference = reference_dr(
        scenario["reward"],
        scenario["action"],
        scenario["pscore"],
        scenario["dist"],
        scenario["q_hat"],
    )
    assert ours == pytest.approx(reference, rel=1e-12)


def test_matches_reference_under_non_uniform_logging(scenario) -> None:
    """Более жёсткий случай: логирующая политика неравномерна, веса разбросаны."""
    rng = np.random.default_rng(99)
    n, n_actions = 2000, 5
    logging_dist = rng.dirichlet(np.ones(n_actions) * 0.5, size=n)
    action = np.array([rng.choice(n_actions, p=logging_dist[i]) for i in range(n)])
    pscore = logging_dist[np.arange(n), action]
    reward = rng.binomial(1, 0.1, size=n).astype(float)
    evaluation_dist = rng.dirichlet(np.ones(n_actions), size=n)

    data = OPEInput(
        reward=reward,
        behavior_pscore=pscore,
        evaluation_pscore=evaluation_dist[np.arange(n), action],
    )
    ours = IPSEstimator().estimate(data)
    reference = reference_ips(reward, action, pscore, evaluation_dist)
    assert ours == pytest.approx(reference, rel=1e-12)
