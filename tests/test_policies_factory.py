import textwrap
import pytest

from explorekit.policies import (
    build_active_policy,
    build_policy,
    build_policy_by_name,
    load_policies_config,
)
from explorekit.policies.epsilon_greedy import EpsilonGreedyPolicy
from explorekit.policies.linucb import LinUCBPolicy


@pytest.fixture
def cfg_file(tmp_path):
    content = textwrap.dedent("""
        seed: 123
        active_policy: linucb
        policies:
          eps:
            type: epsilon_greedy
            n_actions: 5
            epsilon: 0.1
          linucb:
            type: linucb
            n_actions: 5
            context_dim: 3
            alpha: 0.5
            reg: 1.0
    """).strip()
    p = tmp_path / "policies.yaml"
    p.write_text(content)
    return p


def test_build_by_name(cfg_file):
    cfg = load_policies_config(cfg_file)
    policy = build_policy_by_name("eps", cfg)
    assert isinstance(policy, EpsilonGreedyPolicy)
    assert policy.epsilon == 0.1


def test_build_active(cfg_file):
    cfg = load_policies_config(cfg_file)
    policy = build_active_policy(cfg)
    assert isinstance(policy, LinUCBPolicy)
    assert policy.alpha == 0.5


def test_seed_inherited(cfg_file):
    cfg = load_policies_config(cfg_file)
    p1 = build_policy_by_name("eps", cfg)
    p2 = build_policy_by_name("eps", cfg)
                                         
    assert p1.rng.integers(1_000_000) == p2.rng.integers(1_000_000)


def test_unknown_name(cfg_file):
    cfg = load_policies_config(cfg_file)
    with pytest.raises(KeyError, match="not found"):
        build_policy_by_name("nope", cfg)


def test_unknown_type(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(textwrap.dedent("""
        seed: 0
        active_policy: x
        policies:
          x:
            type: not_a_policy
            n_actions: 5
    """).strip())
    cfg = load_policies_config(bad)
    with pytest.raises(ValueError, match="unknown policy type"):
        build_active_policy(cfg)
