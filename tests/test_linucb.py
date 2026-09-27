import numpy as np
import pytest

from explorekit.policies.base import Decision
from explorekit.policies.linucb import LinUCBPolicy


N_ACTIONS = 5
D = 4


@pytest.fixture
def policy():
    return LinUCBPolicy(
        n_actions=N_ACTIONS, context_dim=D,
        alpha=1.0, temperature=1.0, reg=1.0, seed=42,
    )


def test_distribution_sums_to_one(policy):
    x = np.random.default_rng(0).normal(size=(10, D))
    scores = np.zeros((10, N_ACTIONS))
    dist = policy.action_distribution(x, scores)
    assert dist.shape == (10, N_ACTIONS)
    assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-9)
    assert (dist > 0).all()                      


def test_single_round_shape(policy):
    x = np.random.default_rng(0).normal(size=D)
    scores = np.zeros(N_ACTIONS)
    dist = policy.action_distribution(x, scores)
    assert dist.shape == (N_ACTIONS,)
    assert np.isclose(dist.sum(), 1.0)


def test_initial_scores_are_ucb_bonus(policy):
    """После одного наблюдения на action=0 его бонус уменьшается."""
    x = np.ones(D)
    scores_before = policy._ucb_scores(x[None, :])[0]

                                                         
    probs = np.ones(N_ACTIONS) / N_ACTIONS
    d = Decision(
        chosen_item=0,
        propensity=float(probs[0]),          
        probabilities=probs,
        is_exploration=False,
    )
    policy.update(d, reward=1.0, context=x)

    scores_after = policy._ucb_scores(x[None, :])[0]
                                                            
    assert scores_after[0] < scores_before[0]
                                             
    for a in range(1, N_ACTIONS):
        assert np.isclose(scores_after[a], scores_before[a])


def test_mean_reward_moves_theta(policy):
    """После нескольких наград на одном контексте mean-часть скора растёт."""
    x = np.array([1.0, 0.5, -0.3, 0.2])
    probs = np.ones(N_ACTIONS) / N_ACTIONS
    d = Decision(
        chosen_item=0,
        propensity=float(probs[0]),
        probabilities=probs,
        is_exploration=False,
    )

    for _ in range(20):
        policy.update(d, reward=1.0, context=x)

    A_a = policy.A[0]
    b_a = policy.b[0]
    theta_a = np.linalg.solve(A_a, b_a)
    mean_score = float(x @ theta_a)
    assert mean_score > 0.3


def test_update_without_context_uses_last(policy):
    x = np.ones(D)
    scores = np.zeros(N_ACTIONS)
    d = policy.select_action(x, scores)                           
    b_before = policy.b.copy()
    policy.update(d, reward=1.0)                      
    assert not np.allclose(policy.b, b_before)


def test_update_without_context_raises_if_no_select(policy):
    probs = np.ones(N_ACTIONS) / N_ACTIONS
    d = Decision(
        chosen_item=0,
        propensity=float(probs[0]),
        probabilities=probs,
        is_exploration=False,
    )
    with pytest.raises(ValueError, match="context must be provided"):
        policy.update(d, reward=1.0)


def test_candidate_mask(policy):
    x = np.ones(D)
    scores = np.zeros(N_ACTIONS)
    candidates = np.array([1, 3])
    dist = policy.action_distribution(x, scores, candidate_items=candidates)
    assert dist[0] == 0 and dist[2] == 0 and dist[4] == 0
    assert np.isclose(dist[1] + dist[3], 1.0)


def test_temperature_affects_sharpness():
    """Низкая температура -> более острый softmax.

    Важно: у свежего LinUCB все UCB-скоры равны, поэтому сначала
    разводим их разными наградами по действиям.
    """
    p_low = LinUCBPolicy(N_ACTIONS, D, temperature=0.1, seed=0)
    p_high = LinUCBPolicy(N_ACTIONS, D, temperature=10.0, seed=0)
    x = np.ones(D)
    probs = np.ones(N_ACTIONS) / N_ACTIONS

                                                                    
    for a in range(N_ACTIONS):
        d = Decision(a, float(probs[a]), probs, False)
        r = float(a + 1) / N_ACTIONS                            
        p_low.update(d, reward=r, context=x)
        p_high.update(d, reward=r, context=x)

    scores = np.zeros(N_ACTIONS)
    dist_low = p_low.action_distribution(x, scores)
    dist_high = p_high.action_distribution(x, scores)

    ent_low = -(dist_low * np.log(dist_low + 1e-12)).sum()
    ent_high = -(dist_high * np.log(dist_high + 1e-12)).sum()
    assert ent_low < ent_high, f"ent_low={ent_low} vs ent_high={ent_high}"


def test_propensity_matches_probabilities(policy):
    x = np.ones(D)
    scores = np.zeros(N_ACTIONS)
    d = policy.select_action(x, scores)
    assert np.isclose(d.propensity, d.probabilities[d.chosen_item])
    assert d.propensity > 0


def test_ope_temperature_only_affects_action_distribution():
    """temperature влияет на выбор, ope_temperature — только на
    action_distribution (для OPE).

    Важно: чтобы softmax давал разные распределения при разных
    температурах, скоры должны различаться. У свежей политики все
    скоры равны, поэтому сначала "учим" разные награды на разные
    действия.
    """
    policy = LinUCBPolicy(
        N_ACTIONS, D,
        alpha=0.0,
        temperature=0.1,                         
        ope_temperature=10.0,                        
        reg=1.0, seed=0,
    )
    x = np.ones(D)
    probs_uniform = np.ones(N_ACTIONS) / N_ACTIONS

                                                              
    for _ in range(20):
        for a in range(N_ACTIONS):
            r = 1.0 if a == 2 else 0.0
            d = Decision(a, float(probs_uniform[a]), probs_uniform, False)
            policy.update(d, reward=r, context=x)

                                                       
    dist = policy.action_distribution(x, np.zeros(N_ACTIONS))
    assert dist.min() > 0.05, f"OPE-распределение должно быть гладким, dist={dist}"
    assert dist.max() < 0.5, f"OPE-распределение должно быть гладким, dist={dist}"
    assert dist.argmax() == 2, f"action=2 должен доминировать, dist={dist}"

                                                              
    d = policy.select_action(x, np.zeros(N_ACTIONS))
    assert d.probabilities.max() > 0.9, \
        f"выбор должен быть резким, probs={d.probabilities}"
    assert np.isclose(d.propensity, d.probabilities[d.chosen_item])


def test_ope_temperature_defaults_to_temperature():
    p = LinUCBPolicy(N_ACTIONS, D, temperature=2.0, seed=0)
    assert p.ope_temperature == 2.0


def test_invalid_ope_temperature_raises():
    with pytest.raises(ValueError, match="ope_temperature"):
        LinUCBPolicy(N_ACTIONS, D, ope_temperature=0.0)
