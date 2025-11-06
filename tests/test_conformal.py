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
    rng = np.random.default_rng(1234)
    alpha = 0.1
    mu = rng.normal(loc=0.001, scale=0.02, size=400)
    noise = rng.normal(scale=0.02, size=400)
    y = mu + noise

    cal_mu, val_mu = mu[:200], mu[200:]
    cal_y, val_y = y[:200], y[200:]

    sigma_cal = np.full_like(cal_mu, 0.02)
    sigma_val = np.full_like(val_mu, 0.02)
    residuals = np.abs(cal_y - cal_mu) / sigma_cal
    quantile = compute_conformal_quantile(residuals, alpha)
    lower, upper = conformal_interval(val_mu, quantile * sigma_val)
    coverage = np.mean((val_y >= lower) & (val_y <= upper))
    assert 0.82 <= coverage <= 0.97

    realised_error = val_y - val_mu
    normalized_error = np.abs(realised_error) / sigma_val
    p_values = conformal_p_value(residuals, normalized_error)
    assert np.all((p_values >= 0.0) & (p_values <= 1.0))
