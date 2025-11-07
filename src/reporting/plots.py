from __future__ import annotations

"""Plotting utilities for calibration and diagnostics exports."""

import os
from pathlib import Path

import matplotlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_CREATE_SHM", "FALSE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_FORKJOIN_BARRIER", "plain")

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def save_probability_histogram(probabilities: np.ndarray, *, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probs = np.asarray(probabilities, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(probs, bins=20, range=(0.0, 1.0), color="tab:blue", alpha=0.8)
    ax.set_xlabel("Calibrated p_up")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_qq_plot(pit_z: np.ndarray, *, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(pit_z, dtype=float).ravel()
    if values.size == 0:
        values = np.array([0.0])
    sorted_vals = np.sort(values)
    probs = (np.arange(1, sorted_vals.size + 1) - 0.5) / max(sorted_vals.size, 1)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    normal = torch.distributions.Normal(0.0, 1.0)
    theoretical = normal.icdf(torch.from_numpy(probs.astype(np.float64))).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical, sorted_vals, color="tab:purple", alpha=0.7, s=18)
    lims = [min(theoretical.min(), sorted_vals.min()), max(theoretical.max(), sorted_vals.max())]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_xlabel("Theoretical quantiles (N(0,1))")
    ax.set_ylabel("Empirical quantiles (PIT z)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_sign_scatter(mu: np.ndarray, targets: np.ndarray, *, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mu_arr = np.asarray(mu, dtype=float).ravel()
    tgt_arr = np.asarray(targets, dtype=float).ravel()
    pred_sign = np.sign(mu_arr)
    true_sign = np.sign(tgt_arr)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(true_sign, pred_sign, alpha=0.5, color="tab:orange", s=20)
    ax.set_xlabel("sign(y)")
    ax.set_ylabel("sign(μ)")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
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


__all__ = [
    "save_lr_range_plot",
    "save_probability_histogram",
    "save_qq_plot",
    "save_reliability_diagram",
    "save_sign_scatter",
]
