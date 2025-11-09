"""Sharpe ratio scaling validation tests."""  # FIX: ensure annualisation matches specification

import math
import numpy as np

from src.eval.strategy import _annualised_sharpe


def test_annualised_sharpe_matches_manual_computation() -> None:
    """Sharpe computed via helper should align with manual formula."""  # FIX: document expectation

    net_returns = np.array([0.01, -0.005, 0.02, 0.0, -0.01], dtype=float)  # FIX: synthetic net returns
    seconds_per_step = 3600.0  # FIX: one-hour cadence
    freq = (365.0 * 24.0 * 3600.0) / seconds_per_step  # FIX: derive periods per year
    expected = math.sqrt(freq) * net_returns.mean() / net_returns.std(ddof=1)  # FIX: manual Sharpe computation
    observed = _annualised_sharpe(net_returns, seconds_per_step)
    assert math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)  # FIX: ensure negligible numerical drift
