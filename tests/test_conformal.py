from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import pytest

from src.eval.conformal import conformal_interval, conformal_p_value, compute_conformal_quantile


def test_conformal_interval_coverage() -> None:
    """Test conformal interval construction and p-value computation."""  # FIX: add docstring
    rng = np.random.default_rng(1234)
    alpha = 0.1
    mu = rng.normal(loc=0.001, scale=0.02, size=400)
    noise = rng.normal(scale=0.02, size=400)
    y = mu + noise

    cal_mu, val_mu = mu[:200], mu[200:]
    cal_y, val_y = y[:200], y[200:]

    sigma_cal = np.full_like(cal_mu, 0.02)
    sigma_val = np.full_like(val_mu, 0.02)
    from src.eval.conformal import conformal_residuals  # FIX: import residual transformation
    residuals_cal = conformal_residuals(cal_y - cal_mu, sigma_cal, residual="abs")  # FIX: use new residual API
    quantile = compute_conformal_quantile(residuals_cal, alpha)
    lower, upper = conformal_interval(val_mu, sigma_val, quantile, residual="abs")  # FIX: use new conformal_interval API
    coverage = np.mean((val_y >= lower) & (val_y <= upper))
    assert 0.82 <= coverage <= 0.97  # FIX: validate empirical coverage

    residuals_val = conformal_residuals(val_y - val_mu, sigma_val, residual="abs")  # FIX: transform validation residuals
    p_values = conformal_p_value(residuals_cal, residuals_val)  # FIX: compute p-values in transformed space
    assert np.all((p_values >= 0.0) & (p_values <= 1.0))  # FIX: validate p-value domain
