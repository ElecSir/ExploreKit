import numpy as np
import pytest

from explorekit.policies.epsilon_greedy import EpsilonGreedyPolicy


N_ACTIONS = 10


@pytest.fixture
def policy():
    return EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.2, seed=42)


def test_distribution_sums_to_one(policy):
    cand = np.arange(N_ACTIONS)
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    dist = policy.action_distribution(
        context=None, base_scores=scores, candidate_items=cand,
    )
    assert np.isclose(dist.sum(), 1.0, atol=1e-9)
    assert (dist >= 0).all()


def test_non_candidates_get_zero(policy):
    cand = np.array([1, 3, 5])
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    dist = policy.action_distribution(None, scores, cand)              
    non_cand = np.setdiff1d(np.arange(N_ACTIONS), cand)
    assert (dist[non_cand] == 0).all()


def test_propensity_matches_probabilities(policy):
    cand = np.arange(N_ACTIONS)
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    d = policy.select_action(None, scores, cand)                        
    assert np.isclose(d.propensity, d.probabilities[d.chosen_item])
    assert d.propensity > 0


def test_blacklist_excludes_items(policy):
    policy.blacklist = np.array([0, 1])
    cand = np.arange(N_ACTIONS)
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    dist = policy.action_distribution(None, scores, cand)              
    assert dist[0] == 0 and dist[1] == 0


def test_epsilon_zero_is_deterministic():
    p = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.0, seed=0)
    cand = np.arange(N_ACTIONS)
    scores = np.array([0.1] * 5 + [0.9] + [0.1] * 4)
    for _ in range(20):
        d = p.select_action(None, scores, cand)                        
        assert d.chosen_item == 5
        assert d.is_exploration is False


def test_epsilon_one_is_uniform():
    p = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=1.0, seed=0)
    cand = np.arange(N_ACTIONS)
    scores = np.zeros(N_ACTIONS)
    dist = p.action_distribution(None, scores, cand)                   
    assert np.allclose(dist, 1.0 / N_ACTIONS)


def test_chosen_item_rejects_string():
    from explorekit.policies.base import Decision
    with pytest.raises(TypeError):
        Decision(
            chosen_item="item_42",
            propensity=0.5,
            probabilities=np.array([0.5, 0.5]),
            is_exploration=False,
        )


                    

def test_epsilon_clipped_by_max():
    p = EpsilonGreedyPolicy(N_ACTIONS, epsilon=0.9, max_epsilon=0.5, seed=0)
    assert p.epsilon == 0.5


def test_epsilon_clipped_by_min():
    p = EpsilonGreedyPolicy(N_ACTIONS, epsilon=0.01, min_epsilon=0.1, seed=0)
    assert p.epsilon == 0.1


def test_epsilon_in_range_unchanged():
    p = EpsilonGreedyPolicy(
        N_ACTIONS, epsilon=0.3, min_epsilon=0.1, max_epsilon=0.5, seed=0,
    )
    assert p.epsilon == 0.3


def test_invalid_bounds_raise():
    with pytest.raises(ValueError, match="min_epsilon"):
        EpsilonGreedyPolicy(
            N_ACTIONS, epsilon=0.1, min_epsilon=0.5, max_epsilon=0.2,
        )


                          

def test_exploration_subset_best_outside():
    """best ∉ subset -> P(best) = 1 - ε, ε размазывается по subset."""
    p = EpsilonGreedyPolicy(
        n_actions=N_ACTIONS, epsilon=0.4,
        exploration_subset=np.array([0, 1]), seed=0,
    )
    cand = np.arange(N_ACTIONS)
    scores = np.array([0.0, 0.5, 0.9, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    dist = p.action_distribution(None, scores, cand)                   

    assert np.isclose(dist[2], 1.0 - 0.4)
    assert np.isclose(dist[0], 0.2)
    assert np.isclose(dist[1], 0.2)
    assert dist[3] == 0.0
    assert dist[4] == 0.0
    assert np.isclose(dist.sum(), 1.0)


def test_exploration_subset_best_inside():
    """best ∈ subset -> P(best) = 1 - ε + ε/|E|."""
    p = EpsilonGreedyPolicy(
        n_actions=N_ACTIONS, epsilon=0.4,
        exploration_subset=np.array([0, 2]), seed=0,
    )
    cand = np.arange(N_ACTIONS)
    scores = np.array([0.0, 0.5, 0.9, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    dist = p.action_distribution(None, scores, cand)                   

    assert np.isclose(dist[2], 0.4 / 2 + 0.6)
    assert np.isclose(dist[0], 0.2)
    assert dist[1] == 0.0
    assert dist[3] == 0.0
    assert dist[4] == 0.0
    assert np.isclose(dist.sum(), 1.0)


def test_exploration_subset_fallback_when_disjoint():
    """subset ∩ allowed = ∅ -> откат на все allowed."""
    p = EpsilonGreedyPolicy(
        n_actions=N_ACTIONS, epsilon=0.4,
        exploration_subset=np.array([0, 1]),
        seed=0,
    )
    cand = np.array([3, 4])
    scores = np.zeros(N_ACTIONS)
    dist = p.action_distribution(None, scores, cand)                   

    assert dist[0] == 0.0
    assert dist[1] == 0.0
    assert np.isclose(dist[3], 0.2 + 0.6)
    assert np.isclose(dist[4], 0.2)
    assert np.isclose(dist.sum(), 1.0)


def test_exploration_subset_never_zero_probability():
    """propensity > 0 для best."""
    p = EpsilonGreedyPolicy(
        n_actions=N_ACTIONS, epsilon=0.01,
        exploration_subset=np.array([4]), seed=0,
    )
    cand = np.arange(N_ACTIONS)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = p.select_action(None, scores, cand)                            
    assert d.propensity > 0
    assert np.isclose(d.propensity, d.probabilities[d.chosen_item])


                                   

def test_select_action_rejects_batch():
    p = EpsilonGreedyPolicy(N_ACTIONS, epsilon=0.1, seed=0)
    batch_scores = np.zeros((10, N_ACTIONS))
    with pytest.raises(ValueError, match="1D base_scores"):
        p.select_action(None, batch_scores)
