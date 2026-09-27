"""Parquet persistence for DecisionLog batches with daily partitions."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from explorekit.logging.validator import validate_logs

__all__ = ["read_logs_parquet", "write_logs_parquet"]

_JSON_COLUMNS = {
    "context",
    "candidate_items",
    "base_scores",
    "probabilities",
    "policy_params",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_frame(logs: pd.DataFrame | Iterable[Any]) -> pd.DataFrame:
    if isinstance(logs, pd.DataFrame):
        return logs.copy()
    rows = [row.to_record() if hasattr(row, "to_record") else dict(row) for row in logs]
    return pd.DataFrame(rows)


def write_logs_parquet(
    logs: pd.DataFrame | Iterable[Any],
    root_dir: str | Path,
) -> list[Path]:
    """Validate and write logs under ``date=YYYY-MM-DD`` partitions."""

    frame = _to_frame(logs)
    validate_logs(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")

    encoded = frame.copy()
    for column in _JSON_COLUMNS:
        encoded[column] = encoded[column].map(
            lambda value: json.dumps(value, ensure_ascii=False, default=_json_default)
        )

    root = Path(root_dir)
    paths: list[Path] = []
    for date, part in encoded.groupby("date", sort=True):
        partition = root / f"date={date}"
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"part-{uuid.uuid4().hex}.parquet"
        try:
            part.drop(columns=["date"]).to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires pyarrow. Install project requirements first."
            ) from exc
        paths.append(path)
    return paths


def read_logs_parquet(root_dir: str | Path) -> pd.DataFrame:
    """Read all Parquet partitions and restore list/dict columns."""

    root = Path(root_dir)
    files = sorted(root.glob("date=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    try:
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires pyarrow. Install project requirements first."
        ) from exc
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in _JSON_COLUMNS:
        frame[column] = frame[column].map(json.loads)
    return frame
