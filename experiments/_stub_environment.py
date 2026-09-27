"""ВРЕМЕННЫЙ СТАБ симулятора (зона участника 3).

Нужен только для того, чтобы OPE-модуль можно было валидировать до того, как
будет готов настоящий симулятор. Реализует ровно тот контракт, который
ожидает ``explorekit.ope.validation.run_validation``:

    make_replication(seed) -> Replication(ope_input, true_value)

КОГДА УЧАСТНИК 3 СДАСТ СВОЙ МОДУЛЬ, этот файл удаляется, а в
``experiments/ope_synthetic.py`` подменяется одна функция — остальной код
валидации не меняется.

Что внутри: логистическая модель клика P(click|user,item) = sigmoid(theta_a . x),
логирующая политика — softmax с высокой температурой (заведомо стохастическая,
чтобы не нарушать common support), оцениваемая политика — epsilon-greedy.
Истинная ценность политики считается точно, перебором действий, потому что
мы сами задали P(click|user,item).
"""

from __future__ import annotations

import numpy as np

from explorekit.ope.base import OPEInput
from explorekit.ope.validation import Replication

__all__ = ["make_replication"]

N_ACTIONS = 10
N_CONTEXT_FEATURES = 5
                                                                         
                                                                      
                                                                         
                                                           
_INTERCEPT = -5.2
                                                                 
                                                                      
_TRUE_THETA = np.random.default_rng(20240920).normal(
    scale=0.6, size=(N_ACTIONS, N_CONTEXT_FEATURES)
)


def _true_click_prob(context: np.ndarray, action: np.ndarray) -> np.ndarray:
    """P(click | x, a) — скрытая истина симулятора."""
    z = np.einsum("nd,nd->n", context, _TRUE_THETA[action]) + _INTERCEPT
    return 1.0 / (1.0 + np.exp(-z))


def _logging_policy_dist(context: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    """Логирующая политика: softmax по истинному скору с высокой температурой.

    Температура намеренно большая — политика получается близкой к равномерной
    и покрывает все действия, поэтому common support не нарушается.
    """
    scores = context @ _TRUE_THETA.T / temperature
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _epsilon_greedy_dist(base_scores: np.ndarray, epsilon: float) -> np.ndarray:
    """P(best) = 1 - eps + eps/N, P(other) = eps/N."""
    n, n_actions = base_scores.shape
    best = np.argmax(base_scores, axis=1)
    dist = np.full((n, n_actions), epsilon / n_actions)
    dist[np.arange(n), best] += 1.0 - epsilon
    return dist


def make_replication(
    seed: int,
    n_rounds: int = 5000,
    epsilon: float = 0.1,
    base_score_noise: float = 0.3,
    reward_model_noise: float = 0.25,
) -> Replication:
    """Один прогон симулятора.

    Args:
        seed: seed прогона (``np.random.default_rng``, общее правило №3).
        n_rounds: сколько показов сгенерировать.
        epsilon: epsilon оцениваемой политики.
        base_score_noise: шум base ranker-а относительно истины — имитирует
            то, что продовая модель не знает настоящий CTR.
        reward_model_noise: шум reward model для DR — имитирует то, что
            q_hat обучена на конечных данных и неидеальна.

    Returns:
        :class:`Replication` с логами и точной истинной ценностью политики.
    """
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n_rounds, N_CONTEXT_FEATURES))

                                                                
    logging_dist = _logging_policy_dist(context)
    cumulative = np.cumsum(logging_dist, axis=1)
    draws = rng.random((n_rounds, 1))
    chosen = (draws > cumulative).sum(axis=1)
    behavior_pscore = logging_dist[np.arange(n_rounds), chosen]
    reward = rng.binomial(1, _true_click_prob(context, chosen)).astype(float)

                                                    
    noisy_theta_ranker = _TRUE_THETA + rng.normal(
        scale=base_score_noise, size=_TRUE_THETA.shape
    )
    base_scores = context @ noisy_theta_ranker.T
    evaluation_dist = _epsilon_greedy_dist(base_scores, epsilon)
    evaluation_pscore = evaluation_dist[np.arange(n_rounds), chosen]

                                                   
    noisy_theta_reward = _TRUE_THETA + rng.normal(
        scale=reward_model_noise, size=_TRUE_THETA.shape
    )
    q_hat_all = np.zeros((n_rounds, N_ACTIONS))
    for a in range(N_ACTIONS):
        z = context @ noisy_theta_reward[a] + _INTERCEPT
        q_hat_all[:, a] = 1.0 / (1.0 + np.exp(-z))
    q_hat_logged = q_hat_all[np.arange(n_rounds), chosen]
    q_hat_policy = np.sum(evaluation_dist * q_hat_all, axis=1)

                                                                             
    true_value = 0.0
    expected = np.zeros(n_rounds)
    for a in range(N_ACTIONS):
        expected += evaluation_dist[:, a] * _true_click_prob(
            context, np.full(n_rounds, a)
        )
    true_value = float(expected.mean())

    ope_input = OPEInput(
        reward=reward,
        behavior_pscore=behavior_pscore,
        evaluation_pscore=evaluation_pscore,
        q_hat_logged=q_hat_logged,
        q_hat_policy=q_hat_policy,
        policy_name=f"epsilon_greedy(eps={epsilon})",
    )
    return Replication(ope_input=ope_input, true_value=true_value)
