"""Unit tests for conformal prediction intervals per spec."""  # FIX: test module docstring per spec

import numpy as np  # FIX: numpy for array operations
import pytest  # FIX: pytest framework

from src.eval.conformal import (  # FIX: import conformal utilities
    compute_conformal_quantile,
    conformal_interval,
    conformal_p_value,
    conformal_residuals,
)


def test_conformal_coverage_abs_residuals():  # FIX: test coverage for absolute residuals per spec
    """Test that conformal intervals achieve empirical coverage within ±2% for absolute residuals."""  # FIX: test docstring per spec
    np.random.seed(42)  # FIX: set seed for reproducibility per spec
    n_calib = 1000  # FIX: calibration sample size
    n_val = 500  # FIX: validation sample size
    alpha = 0.10  # FIX: miscoverage level per spec
    target_coverage = 1.0 - alpha  # FIX: target coverage 90%

    # FIX: Generate synthetic Gaussian data
    y_calib = np.random.randn(n_calib)  # FIX: calibration targets
    mu_calib = np.zeros(n_calib)  # FIX: calibration predictions (zero mean)
    sigma_calib = np.ones(n_calib)  # FIX: calibration sigmas (unit variance)

    y_val = np.random.randn(n_val)  # FIX: validation targets
    mu_val = np.zeros(n_val)  # FIX: validation predictions
    sigma_val = np.ones(n_val)  # FIX: validation sigmas

    # FIX: Compute calibration residuals
    calib_error = y_calib - mu_calib  # FIX: calibration errors
    calib_residuals = conformal_residuals(calib_error, sigma_calib, residual="abs")  # FIX: absolute residuals per spec

    # FIX: Compute conformal quantile
    quantile = compute_conformal_quantile(calib_residuals, alpha)  # FIX: get quantile per spec

    # FIX: Compute conformal intervals
    lower, upper = conformal_interval(mu_val, sigma_val, quantile, residual="abs")  # FIX: intervals per spec

    # FIX: Check empirical coverage
    coverage = np.mean((y_val >= lower) & (y_val <= upper))  # FIX: empirical coverage

    # FIX: Assert coverage within ±2% tolerance per spec
    tolerance = 0.02  # FIX: ±2% tolerance per spec
    assert abs(coverage - target_coverage) <= tolerance, (
        f"Conformal coverage {coverage:.4f} not within [{target_coverage - tolerance:.4f}, "
        f"{target_coverage + tolerance:.4f}]"
    )  # FIX: assert coverage tolerance per spec


def test_conformal_coverage_std_gauss_residuals():  # FIX: test coverage for standardized Gaussian residuals per spec
    """Test that conformal intervals achieve empirical coverage for std_gauss residuals."""  # FIX: test docstring per spec
    np.random.seed(42)  # FIX: set seed for reproducibility per spec
    n_calib = 1000  # FIX: calibration sample size
    n_val = 500  # FIX: validation sample size
    alpha = 0.10  # FIX: miscoverage level per spec
    target_coverage = 1.0 - alpha  # FIX: target coverage 90%

    # FIX: Generate synthetic heteroscedastic Gaussian data
    y_calib = np.random.randn(n_calib) * (1.0 + 0.5 * np.random.randn(n_calib))  # FIX: heteroscedastic calibration targets
    mu_calib = np.zeros(n_calib)  # FIX: calibration predictions
    sigma_calib = 1.0 + 0.5 * np.abs(np.random.randn(n_calib))  # FIX: calibration sigmas (heteroscedastic)

    y_val = np.random.randn(n_val) * (1.0 + 0.5 * np.random.randn(n_val))  # FIX: heteroscedastic validation targets
    mu_val = np.zeros(n_val)  # FIX: validation predictions
    sigma_val = 1.0 + 0.5 * np.abs(np.random.randn(n_val))  # FIX: validation sigmas

    # FIX: Compute calibration residuals
    calib_error = y_calib - mu_calib  # FIX: calibration errors
    calib_residuals = conformal_residuals(calib_error, sigma_calib, residual="std_gauss")  # FIX: std_gauss residuals per spec

    # FIX: Compute conformal quantile
    quantile = compute_conformal_quantile(calib_residuals, alpha)  # FIX: get quantile per spec

    # FIX: Compute conformal intervals
    lower, upper = conformal_interval(mu_val, sigma_val, quantile, residual="std_gauss")  # FIX: intervals per spec

    # FIX: Check empirical coverage
    coverage = np.mean((y_val >= lower) & (y_val <= upper))  # FIX: empirical coverage

    # FIX: Assert coverage within ±2% tolerance per spec
    tolerance = 0.02  # FIX: ±2% tolerance per spec
    assert abs(coverage - target_coverage) <= tolerance, (
        f"Conformal coverage {coverage:.4f} not within [{target_coverage - tolerance:.4f}, "
        f"{target_coverage + tolerance:.4f}]"
    )  # FIX: assert coverage tolerance per spec


def test_conformal_p_values():  # FIX: test conformal p-values computation per spec
    """Test conformal p-value computation logic."""  # FIX: test docstring
    np.random.seed(42)  # FIX: set seed for reproducibility
    n_calib = 100  # FIX: calibration sample size

    # FIX: Generate synthetic calibration residuals
    calib_residuals = np.abs(np.random.randn(n_calib))  # FIX: absolute residuals

    # FIX: Generate test residuals
    test_residuals = np.array([0.5, 1.5, 2.5])  # FIX: test residuals

    # FIX: Compute conformal p-values
    p_values = conformal_p_value(calib_residuals, test_residuals)  # FIX: p-values per spec

    # FIX: Assert p-values are in valid range
    assert np.all((p_values >= 0.0) & (p_values <= 1.0)), "Conformal p-values must be in [0, 1]"  # FIX: assert valid range

    # FIX: Assert p-values are monotonically increasing for increasing residuals
    assert p_values[0] <= p_values[1] <= p_values[2], "Conformal p-values should increase with residual magnitude"  # FIX: assert monotonicity


# FIX: Export test functions
__all__ = [
    "test_conformal_coverage_abs_residuals",
    "test_conformal_coverage_std_gauss_residuals",
    "test_conformal_p_values",
]  # FIX: module exports
