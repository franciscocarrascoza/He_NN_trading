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

from src.eval.diagnostics import (
    DieboldMarianoResult,
    binomial_test_pvalue,
    diebold_mariano,
    ljung_box,
    mincer_zarnowitz,
    probability_calibration_metrics,
    runs_test,
)


def test_diebold_mariano_detects_skill() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(size=300)
    skilled_forecast = y_true + rng.normal(scale=0.1, size=300)
    noisy_benchmark = rng.normal(scale=1.5, size=300)
    result = diebold_mariano(y_true, skilled_forecast, noisy_benchmark, horizon=1, loss="mse")
    assert isinstance(result, DieboldMarianoResult)
    assert 0.0 <= result.p_value <= 0.1
    assert result.mean_loss_diff < 0.0


def test_probability_metrics_basic_values() -> None:
    probs = np.linspace(0.1, 0.9, 20)
    returns = np.where(probs > 0.5, 0.01, -0.01)
    metrics = probability_calibration_metrics(returns, probs)
    assert 0.0 <= metrics.brier <= 1.0
    assert metrics.bin_edges.size == 11
    assert metrics.brier_uncertainty >= 0.0
    assert metrics.brier_resolution >= 0.0
    assert metrics.brier_reliability >= 0.0
    reconstructed = metrics.brier_reliability - metrics.brier_resolution + metrics.brier_uncertainty
    assert pytest.approx(metrics.brier, rel=1e-3, abs=1e-3) == reconstructed


def test_mincer_zarnowitz_stats() -> None:
    rng = np.random.default_rng(11)
    y_true = rng.normal(size=200)
    forecast = y_true + rng.normal(scale=0.2, size=200)
    mz = mincer_zarnowitz(y_true, forecast)
    assert np.isfinite(mz.intercept)
    assert np.isfinite(mz.slope)
    assert np.isfinite(mz.p_value)


def test_runs_and_ljungbox_random_noise() -> None:
    rng = np.random.default_rng(19)
    residuals = rng.normal(size=300)
    runs_p = runs_test(residuals)
    _, ljung_p = ljung_box(residuals)
    assert 0.0 <= runs_p <= 1.0
    assert 0.0 <= ljung_p <= 1.0


def test_binomial_two_sided_mid_p() -> None:
    p_value = binomial_test_pvalue(6, 10)
    assert 0.0 <= p_value <= 1.0
