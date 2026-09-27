"""Unified experiment result contract and persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from explorekit.ope import ReliabilityReport

__all__ = [
    "ExperimentResult",
    "ReliabilityReport",
    "load_experiment_results",
    "save_experiment_result",
]


@dataclass
class ExperimentResult:
    """One end-to-end experiment in the common reporting format."""

    experiment_name: str
    goal: str
    preset_name: str
    logging_policy: str
    evaluation_policy: str
    n_rounds: int
    seed: int
    parameters: dict[str, Any]
    true_ctr: float | None
    expected_ctr: float | None
    expected_ctr_estimator: str | None
    expected_ci_low: float | None
    expected_ci_high: float | None
    ips_estimate: float | None
    ips_ci_low: float | None
    ips_ci_high: float | None
    snips_estimate: float | None
    snips_ci_low: float | None
    snips_ci_high: float | None
    dr_estimate: float | None
    dr_ci_low: float | None
    dr_ci_high: float | None
    ess: float
    ess_ratio: float
    reliability: str
    reliability_explanation: str
    exploration_share: float | None = None
    cold_item_speedup: float | None = None
    ctr_cost: float | None = None
    conclusion: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        return cls(**data)


def save_experiment_result(
    result: ExperimentResult,
    output_dir: str | Path,
) -> Path:
    """Save one result as JSON for the dashboard and final report."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_policy = result.evaluation_policy.replace("/", "_").replace(" ", "_")
    path = root / (
        f"{result.preset_name}_{safe_policy}_seed{result.seed}_{timestamp}.json"
    )
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_experiment_results(output_dir: str | Path) -> pd.DataFrame:
    """Load all saved JSON results into a flat DataFrame."""

    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)
