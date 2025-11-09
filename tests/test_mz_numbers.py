"""Mincer–Zarnowitz regression sanity tests."""  # FIX: ensure OLS diagnostics accurate

import numpy as np

from src.eval.diagnostics import mincer_zarnowitz


def test_mincer_zarnowitz_recovers_unit_slope() -> None:
    """Synthetic data with unit slope should return near-ideal coefficients."""  # FIX: describe expectation

    rng = np.random.default_rng(123)
    mu = np.linspace(-1.0, 1.0, 512)
    noise = rng.normal(scale=0.01, size=mu.size)
    y = mu + noise
    result = mincer_zarnowitz(y, mu)
    assert abs(result.intercept) < 0.01  # FIX: intercept close to zero
    assert abs(result.slope - 1.0) < 0.01  # FIX: slope close to unity
    assert result.p_value > 0.05  # FIX: cannot reject ideal calibration
