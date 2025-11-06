from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.eval.diagnostics import CalibrationMetrics


def save_reliability_diagram(
    metrics: CalibrationMetrics,
    *,
    output_path: Path,
    title: str = "Reliability Diagram",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    mask = metrics.bin_counts > 0
    confidences = metrics.bin_confidence[mask]
    accuracies = metrics.bin_accuracy[mask]
    sizes = metrics.bin_counts[mask]
    if confidences.size:
        size_norm = sizes / sizes.max()
        scatter_sizes = 50 + 150 * size_norm
        ax.scatter(confidences, accuracies, s=scatter_sizes, c="tab:blue", alpha=0.7, label="Observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_lr_range_plot(lr_values: np.ndarray, losses: list[float], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lr_values, losses, marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title("LR Range Test")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = ["save_lr_range_plot", "save_reliability_diagram"]
