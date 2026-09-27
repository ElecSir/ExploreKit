from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from explorekit.logging import DecisionLog, read_logs_parquet, write_logs_parquet
from explorekit.policies import Decision

pytest.importorskip("pyarrow")


def test_parquet_roundtrip(tmp_path) -> None:
    decision = Decision(
        chosen_item=1,
        propensity=0.6,
        probabilities=[0.4, 0.6],
        is_exploration=False,
        policy_name="epsilon_greedy",
    )
    log = DecisionLog.from_decision(
        decision=decision,
        timestamp=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
        request_id="req-1",
        user_segment="a",
        context=[1.0, 2.0],
        candidate_items=[0, 1],
        base_scores=[0.2, 0.8],
        policy_params={"epsilon": 0.1},
        reward=1,
        is_cold_item=True,
    )
    paths = write_logs_parquet([log], tmp_path)
    assert len(paths) == 1
    assert "date=2026-09-20" in str(paths[0])
    restored = read_logs_parquet(tmp_path)
    assert len(restored) == 1
    assert restored.iloc[0]["candidate_items"] == [0, 1]
    assert restored.iloc[0]["policy_params"] == {"epsilon": 0.1}
    assert isinstance(restored.iloc[0]["timestamp"], pd.Timestamp)
