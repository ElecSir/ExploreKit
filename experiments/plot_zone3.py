"""Графики trade-off exploration и сверка OPE.

Запуск:
    python experiments/plot_zone3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cold_item_experiment import run_cold_item_experiment

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"
DATA = ROOT / "report" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Графики exploration / OPE")
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    return parser.parse_args()


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_frames(frame: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    share_pct = frame["exploration_share"].astype(float) * 100
    ctr_pct = frame["true_ctr"].astype(float) * 100

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(share_pct, ctr_pct, marker="o")
    for _, row in frame.iterrows():
        ax.annotate(str(row["preset"]), (row["exploration_share"] * 100, row["true_ctr"] * 100))
    ax.set_xlabel("Exploration share, %")
    ax.set_ylabel("True CTR, %")
    ax.set_title("Exploration vs CTR")
    ax.grid(True, alpha=0.3)
    paths.append(_save(fig, "exploration_vs_ctr.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(share_pct, frame["speedup"].astype(float), marker="o")
    for _, row in frame.iterrows():
        ax.annotate(str(row["preset"]), (row["exploration_share"] * 100, row["speedup"]))
    ax.set_xlabel("Exploration share, %")
    ax.set_ylabel("Cold-item speedup")
    ax.set_title("Exploration vs cold-item speedup")
    ax.grid(True, alpha=0.3)
    paths.append(_save(fig, "exploration_vs_speedup.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = range(len(frame))
    width = 0.18
    series = [
        ("True", frame["true_ctr"].astype(float) * 100),
        ("IPS", frame["ips_estimate"].astype(float) * 100),
        ("SNIPS", frame["snips_estimate"].astype(float) * 100),
        ("DR", frame["dr_estimate"].astype(float) * 100),
    ]
    for i, (label, values) in enumerate(series):
        offset = (i - 1.5) * width
        ax.bar([p + offset for p in x], values, width=width, label=label)
    ax.set_xticks(list(x), list(frame["preset"]))
    ax.set_ylabel("CTR, %")
    ax.set_title("True CTR vs OPE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    paths.append(_save(fig, "true_vs_ope.png"))
    return paths


def main() -> None:
    args = parse_args()
    frame = run_cold_item_experiment(
        n_rounds=args.n_rounds,
        with_ope=True,
        n_bootstrap=args.n_bootstrap,
    )
    DATA.mkdir(parents=True, exist_ok=True)
    csv_path = DATA / "presets.csv"
    frame.to_csv(csv_path, index=False)
    figure_paths = plot_frames(frame)
    print(frame.to_string(index=False))
    print("csv:", csv_path)
    print("figures:", ", ".join(path.name for path in figure_paths))


if __name__ == "__main__":
    main()
