"""Conformal coverage stability tests."""  # FIX: validate residual-driven coverage

import numpy as np

from src.eval.conformal import (
    compute_conformal_quantile,
    conformal_interval,
    conformal_residuals,
)


def _simulate_gaussian(alpha: float, residual: str) -> float:
    """Helper to estimate coverage for a Gaussian oracle."""  # FIX: document helper intent

    rng = np.random.default_rng(42)  # FIX: deterministic randomness
    sigma = 0.2  # FIX: known volatility
    cal_size = 4096  # FIX: ample calibration data
    val_size = 2048  # FIX: evaluation sample size
    mu_cal = np.zeros(cal_size)
    mu_val = np.zeros(val_size)
    sigma_cal = np.full(cal_size, sigma)
    sigma_val = np.full(val_size, sigma)
    y_cal = rng.normal(loc=0.0, scale=sigma, size=cal_size)
    y_val = rng.normal(loc=0.0, scale=sigma, size=val_size)
    cal_residuals = conformal_residuals(y_cal - mu_cal, sigma_cal, residual=residual)
    quantile = compute_conformal_quantile(cal_residuals, alpha)
    val_residuals = conformal_residuals(y_val - mu_val, sigma_val, residual=residual)
    lower, upper = conformal_interval(mu_val, sigma_val, quantile, residual=residual)
    coverage = np.mean((y_val >= lower) & (y_val <= upper))
    assert np.isfinite(val_residuals).all()  # FIX: ensure transformed residuals remain finite
    return float(coverage)


def test_conformal_abs_residual_coverage() -> None:
    """Absolute residual conformal should meet nominal coverage within tolerance."""  # FIX: describe assertion

    coverage = _simulate_gaussian(alpha=0.1, residual="abs")
    assert abs(coverage - 0.9) <= 0.02  # FIX: enforce ±2% tolerance


def test_conformal_std_gauss_residual_coverage() -> None:
    """Log-standardised residual conformal should also meet nominal coverage."""  # FIX: describe assertion

    coverage = _simulate_gaussian(alpha=0.1, residual="std_gauss")
    assert abs(coverage - 0.9) <= 0.02  # FIX: enforce ±2% tolerance
