import numpy as np
import pytest

from explorekit.policies.base import Decision
from explorekit.policies.thompson import ThompsonSamplingPolicy


N_ACTIONS = 5
D = 4


@pytest.fixture
def policy():
    return ThompsonSamplingPolicy(
        n_actions=N_ACTIONS, context_dim=D,
        mc_samples=100, reg=1.0, seed=42,
    )


def make_decision(chosen=0):
    probs = np.ones(N_ACTIONS) / N_ACTIONS
    return Decision(chosen, float(probs[chosen]), probs, False)


def test_distribution_sums_to_one(policy):
    x = np.random.default_rng(0).normal(size=(10, D))
    scores = np.zeros((10, N_ACTIONS))
    dist = policy.action_distribution(x, scores)
    assert dist.shape == (10, N_ACTIONS)
    assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-9)
    assert (dist >= 0).all()


def test_single_round_shape(policy):
    x = np.random.default_rng(0).normal(size=D)
    dist = policy.action_distribution(x, np.zeros(N_ACTIONS))
    assert dist.shape == (N_ACTIONS,)
    assert np.isclose(dist.sum(), 1.0)


def test_propensity_matches_probabilities(policy):
    x = np.ones(D)
    d = policy.select_action(x, np.zeros(N_ACTIONS))
    assert np.isclose(d.propensity, d.probabilities[d.chosen_item])
    assert d.propensity > 0


def test_candidate_mask(policy):
    x = np.ones(D)
    candidates = np.array([1, 3])
    dist = policy.action_distribution(x, np.zeros(N_ACTIONS), candidate_items=candidates)
    assert dist[0] == 0 and dist[2] == 0 and dist[4] == 0
    assert np.isclose(dist[1] + dist[3], 1.0)


def test_update_changes_state(policy):
    x = np.ones(D)
    d = make_decision(0)
    A_before = policy.A[0].copy()
    policy.update(d, reward=1.0, context=x)
    assert not np.allclose(policy.A[0], A_before)
    assert not np.allclose(policy.b[0], 0)


def test_mc_propensity_is_monte_carlo(policy):
    """При большом mc_samples распределение должно быть близко к 1/n_actions
    в начале, когда все действия неразличимы."""
    x = np.ones(D)
    dist = policy.action_distribution(x, np.zeros(N_ACTIONS))
                                                                  
    expected = 1.0 / N_ACTIONS
    assert np.allclose(dist, expected, atol=0.1), f"dist={dist}"


def test_convergence_to_best_action():
    """После многих наград на action=2 и нулей на остальных TS должен
    предпочитать action=2.

    Важно: обновляем ВСЕ действия, а не только action=2. Иначе у остальных
    остаётся широкая prior-дисперсия, и они «выстреливают» в сэмплах.
    """
    policy = ThompsonSamplingPolicy(N_ACTIONS, D, mc_samples=300, seed=0)
    x = np.ones(D)
    probs = np.ones(N_ACTIONS) / N_ACTIONS

                                                                            
    for _ in range(50):
        for a in range(N_ACTIONS):
            reward = 1.0 if a == 2 else 0.0
            d = Decision(a, float(probs[a]), probs, False)
            policy.update(d, reward=reward, context=x)

    dist = policy.action_distribution(x, np.zeros(N_ACTIONS))

                                  
    assert dist[2] > 0.5, f"dist={dist}, expected action 2 to dominate"
    assert dist[2] == dist.max(), f"dist={dist}, expected action 2 to be max"


def test_update_without_context_uses_last(policy):
    x = np.ones(D)
    d = policy.select_action(x, np.zeros(N_ACTIONS))
    b_before = policy.b.copy()
    policy.update(d, reward=1.0)
    assert not np.allclose(policy.b, b_before)


def test_update_without_context_raises_if_no_select(policy):
    d = make_decision(0)
    with pytest.raises(ValueError, match="context must be provided"):
        policy.update(d, reward=1.0)


def test_uncertainty_shrinks_with_data():
    """После наблюдений posterior для действия сужается."""
    policy = ThompsonSamplingPolicy(N_ACTIONS, D, mc_samples=200, seed=0)
    x = np.ones(D)
    probs = np.ones(N_ACTIONS) / N_ACTIONS
    d = Decision(0, float(probs[0]), probs, False)

                                                                      
    var_before = float(x @ np.linalg.solve(policy.A[0], x))
    assert np.isclose(var_before, 4.0)

                      
    for _ in range(20):
        policy.update(d, reward=1.0, context=x)

    var_after = float(x @ np.linalg.solve(policy.A[0], x))
    assert var_after < var_before, "дисперсия должна уменьшаться"
