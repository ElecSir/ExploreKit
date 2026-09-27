"""Streamlit dashboard for saved ExploreKit experiment results."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorekit.results import load_experiment_results              

st.set_page_config(page_title="ExploreKit", layout="wide")
st.title("ExploreKit — Exploration & OPE")

results_dir = Path(os.getenv("EXPLOREKIT_RESULTS_DIR", ROOT / "outputs" / "results"))
frame = load_experiment_results(results_dir)

if frame.empty:
    st.info(
        "No saved results yet. Run: "
        "python experiments/run_experiment.py --config configs/moderate.yaml"
    )
    st.stop()

if "created_at" in frame.columns:
    frame = frame.sort_values("created_at")
latest = frame.groupby("preset_name", as_index=False).tail(1).copy()


def pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def ci_text(row: pd.Series) -> str:
    low = row.get("expected_ci_low")
    high = row.get("expected_ci_high")
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "N/A"
    return f"[{float(low) * 100:.2f}%, {float(high) * 100:.2f}%]"


st.subheader("Overview")
overview_rows = []
for _, row in latest.iterrows():
    overview_rows.append(
        {
            "Preset": row.get("preset_name"),
            "Policy": row.get("evaluation_policy"),
            "Expected CTR": pct(row.get("expected_ctr")),
            "Estimator": row.get("expected_ctr_estimator"),
            "True CTR": pct(row.get("true_ctr")),
            "95% CI": ci_text(row),
            "Exploration share": pct(row.get("exploration_share")),
            "Cold-item speedup": (
                "N/A"
                if pd.isna(row.get("cold_item_speedup"))
                else f"{float(row.get('cold_item_speedup')):.2f}x"
            ),
            "Reliability": row.get("reliability"),
        }
    )
st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

st.subheader("Exploration vs Exploitation")
chart = latest[["preset_name", "exploration_share", "true_ctr"]].copy()
chart = chart.dropna(subset=["exploration_share", "true_ctr"])
if not chart.empty:
    chart["exploration_share_pct"] = chart["exploration_share"] * 100
    chart["true_ctr_pct"] = chart["true_ctr"] * 100
    st.scatter_chart(
        chart,
        x="exploration_share_pct",
        y="true_ctr_pct",
        size=90,
    )
else:
    st.caption("Exploration/CTR metrics are not available yet.")

cold_chart = latest[["preset_name", "exploration_share", "cold_item_speedup"]].copy()
cold_chart = cold_chart.dropna(subset=["exploration_share", "cold_item_speedup"])
if not cold_chart.empty:
    cold_chart["exploration_share_pct"] = cold_chart["exploration_share"] * 100
    st.scatter_chart(
        cold_chart,
        x="exploration_share_pct",
        y="cold_item_speedup",
        size=90,
    )
else:
    st.caption(
        "Cold-item speedup is missing from saved experiment results."
    )

st.subheader("OPE comparison")
ope_rows = []
for _, row in latest.iterrows():
    for estimator in ("ips", "snips", "dr"):
        estimate = row.get(f"{estimator}_estimate")
        if estimate is None or pd.isna(estimate):
            continue
        ope_rows.append(
            {
                "Preset": row.get("preset_name"),
                "Policy": row.get("evaluation_policy"),
                "Estimator": estimator.upper(),
                "Estimate": pct(estimate),
                "True CTR": pct(row.get("true_ctr")),
                "ESS": round(float(row.get("ess", 0.0)), 1),
                "Reliability": row.get("reliability"),
            }
        )
st.dataframe(pd.DataFrame(ope_rows), use_container_width=True, hide_index=True)

st.subheader("Reliability diagnostics")
preset = st.selectbox("Preset", latest["preset_name"].tolist())
selected = latest[latest["preset_name"] == preset].iloc[-1]
st.write(selected.get("reliability_explanation", "No diagnostics available."))

diagnostics = selected.get("diagnostics")
if isinstance(diagnostics, dict):
    diag_table = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in diagnostics.items()]
    )
    st.dataframe(diag_table, use_container_width=True, hide_index=True)
