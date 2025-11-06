from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def compute_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1-D array.")
    if residuals.size == 0:
        raise ValueError("At least one calibration residual is required.")
    alpha = float(np.clip(alpha, 1e-6, 0.999))
    sorted_residuals = np.sort(np.asarray(residuals, dtype=float))
    n = sorted_residuals.size
    rank = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(sorted_residuals[rank])


def conformal_interval(mu: np.ndarray, quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    return mu - quantile, mu + quantile


def conformal_p_value(calibration_residuals: np.ndarray, realised_error: np.ndarray) -> np.ndarray:
    calibration_residuals = np.abs(np.asarray(calibration_residuals, dtype=float))
    realised_error = np.abs(np.asarray(realised_error, dtype=float))
    n = calibration_residuals.size
    if n == 0:
        raise ValueError("Calibration residuals cannot be empty when computing conformal p-values.")
    comparison = realised_error[:, None] <= calibration_residuals[None, :]
    counts = comparison.sum(axis=1)
    p_values = (1.0 + counts) / (1.0 + n)
    return p_values


__all__ = ["compute_conformal_quantile", "conformal_interval", "conformal_p_value"]
