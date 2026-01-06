"""Visualization utilities.

Plots are saved to outputs/ as PNG files using default matplotlib styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def plot_coverage(logs_by_strategy: Mapping[str, pd.DataFrame]) -> Path:
    """
    Plot high-risk coverage over time for multiple strategies.

    Expected per-strategy DataFrame columns:
      - step
      - high_risk_coverage

    Saves: outputs/coverage.png
    """
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "coverage.png"

    plt.figure(figsize=(10, 5))
    for name, df in logs_by_strategy.items():
        if len(df) == 0:
            continue
        plt.plot(df["step"], df["high_risk_coverage"], label=name)

    plt.xlabel("Step")
    plt.ylabel("High-risk coverage")
    plt.title("High-risk coverage over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_distance(logs_by_strategy: Mapping[str, pd.DataFrame]) -> Path:
    """
    Plot total distance over time for multiple strategies.

    Expected per-strategy DataFrame columns:
      - step
      - distance

    Saves: outputs/distance.png
    """
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "distance.png"

    plt.figure(figsize=(10, 5))
    for name, df in logs_by_strategy.items():
        if len(df) == 0:
            continue
        plt.plot(df["step"], df["distance"], label=name)

    plt.xlabel("Step")
    plt.ylabel("Total distance")
    plt.title("Total distance over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
