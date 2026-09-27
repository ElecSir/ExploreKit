"""OPE-модуль ExploreKit (зона ответственности участника 2).

Публичный API::

    from explorekit.ope import (
        OPEInput, OPEResult, build_ope_input,
        IPSEstimator, SNIPSEstimator, DREstimator,
        RewardModel, bootstrap_ci, compute_reliability,
    )
"""

from explorekit.ope.base import (
    ActionDistributionFn,
    OPEEstimator,
    OPEInput,
    OPEResult,
    build_ope_input,
)
from explorekit.ope.bootstrap import bootstrap_ci
from explorekit.ope.diagnostics import (
    ReliabilityReport,
    ReliabilityStatus,
    compute_reliability,
)
from explorekit.ope.dr import DREstimator, RewardModel, split_for_reward_model
from explorekit.ope.ips import IPSEstimator, clip_weights
from explorekit.ope.snips import SNIPSEstimator

__all__ = [
    "ActionDistributionFn",
    "OPEEstimator",
    "OPEInput",
    "OPEResult",
    "build_ope_input",
    "bootstrap_ci",
    "ReliabilityReport",
    "ReliabilityStatus",
    "compute_reliability",
    "DREstimator",
    "RewardModel",
    "split_for_reward_model",
    "IPSEstimator",
    "clip_weights",
    "SNIPSEstimator",
]
