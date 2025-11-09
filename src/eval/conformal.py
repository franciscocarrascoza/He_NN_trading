"""Conformal calibration utilities supporting multiple residual families."""  # FIX: document conformal helpers

from __future__ import annotations  # FIX: modern typing compatibility

import math
from typing import Literal, Tuple

import numpy as np

ResidualKind = Literal["abs", "std_gauss"]  # FIX: enumerate supported residual encodings


def conformal_residuals(
    realised_error: np.ndarray,
    sigma: np.ndarray,
    *,
    residual: ResidualKind,
) -> np.ndarray:
    """Transform residuals according to the configured conformal family."""  # FIX: describe transformation logic

    abs_error = np.abs(np.asarray(realised_error, dtype=float))  # FIX: base absolute error
    safe_sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)  # FIX: prevent division by zero
    if residual == "abs":  # FIX: plain absolute residuals in target space
        return abs_error  # FIX: return directly for absolute calibration
    if residual == "std_gauss":  # FIX: Gaussianised residual encoding
        standardized = abs_error / safe_sigma  # FIX: normalise by predictive sigma
        return np.log(np.clip(standardized, 1e-6, None))  # FIX: log-standardised residuals per spec
    raise ValueError(f"Unsupported residual type: {residual}")  # FIX: defensive programming


def compute_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Return the conformal quantile for a target miscoverage level."""  # FIX: clarify quantile semantics

    if residuals.ndim != 1:  # FIX: ensure simple vector input
        raise ValueError("residuals must be a 1-D array.")  # FIX: consistent validation messaging
    if residuals.size == 0:  # FIX: guard against empty calibration sets
        raise ValueError("At least one calibration residual is required.")  # FIX: avoid invalid quantile calc
    alpha = float(np.clip(alpha, 1e-6, 0.999))  # FIX: stabilise boundary alpha values
    sorted_residuals = np.sort(np.asarray(residuals, dtype=float))  # FIX: ensure numeric sort order
    n = sorted_residuals.size  # FIX: reuse length for rank calculation
    rank = int(math.ceil((n + 1) * (1.0 - alpha))) - 1  # FIX: conformal rank per ICP recipe
    rank = min(max(rank, 0), n - 1)  # FIX: clamp rank to valid bounds
    return float(sorted_residuals[rank])  # FIX: quantile output as float


def conformal_interval(
    mu: np.ndarray,
    sigma: np.ndarray,
    quantile: float,
    *,
    residual: ResidualKind,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct conformal intervals consistent with the residual encoding."""  # FIX: explain interval construction

    if residual == "abs":  # FIX: absolute residual intervals in raw units
        return mu - quantile, mu + quantile  # FIX: symmetric absolute interval
    if residual == "std_gauss":  # FIX: invert log-standardised quantile into scale multiplier
        scale = np.exp(quantile)  # FIX: convert log-quantile to multiplicative scale
        return mu - scale * sigma, mu + scale * sigma  # FIX: rescale by predictive sigma
    raise ValueError(f"Unsupported residual type: {residual}")  # FIX: protect against invalid configs


def conformal_p_value(
    calibration_residuals: np.ndarray,
    realised_residuals: np.ndarray,
) -> np.ndarray:
    """Compute conformal p-values via empirical exceedance counts."""  # FIX: clarify statistical meaning

    calibration_residuals = np.asarray(calibration_residuals, dtype=float)  # FIX: normalise calibration inputs
    realised_residuals = np.asarray(realised_residuals, dtype=float)  # FIX: cast realised residuals
    n = calibration_residuals.size  # FIX: track calibration size
    if n == 0:  # FIX: guard empty calibration set
        raise ValueError("Calibration residuals cannot be empty when computing conformal p-values.")  # FIX: explicit error
    comparison = realised_residuals[:, None] <= calibration_residuals[None, :]  # FIX: vectorised comparison
    counts = comparison.sum(axis=1)  # FIX: count calibration exceedances
    p_values = (1.0 + counts) / (1.0 + n)  # FIX: smoothed conformal p-value estimate
    return p_values  # FIX: return numpy array of p-values


__all__ = [
    "ResidualKind",
    "compute_conformal_quantile",
    "conformal_interval",
    "conformal_p_value",
    "conformal_residuals",
]  # FIX: expose updated API surface
