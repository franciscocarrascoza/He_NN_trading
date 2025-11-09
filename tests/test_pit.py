"""PIT z-score calibration tests."""  # FIX: ensure PIT diagnostics behave

import math
import numpy as np

from src.utils.utils import pit_zscore


def _ks_pvalue_normal(samples: np.ndarray) -> float:
    """Approximate Kolmogorov–Smirnov p-value against N(0,1)."""  # FIX: helper for KS approximation

    values = np.sort(samples)
    n = values.size
    if n == 0:
        return float("nan")
    cdf = 0.5 * (1.0 + np.erf(values / math.sqrt(2.0)))
    empirical = np.arange(1, n + 1) / n
    d_plus = np.max(empirical - cdf)
    d_minus = np.max(cdf - (np.arange(0, n) / n))
    d = max(d_plus, d_minus)
    if d <= 0:
        return 1.0
    # Smirnov approximation truncated at first few terms.
    p_value = 0.0
    for k in range(1, 6):
        p_value += (-1) ** (k - 1) * math.exp(-2.0 * (k**2) * (d**2) * n)
    return max(0.0, min(1.0, 2.0 * p_value))


def test_pit_zscores_match_standard_normal() -> None:
    """Synthetic standard normal inputs should not trigger KS rejection."""  # FIX: describe expectation

    rng = np.random.default_rng(42)
    y = rng.normal(loc=0.0, scale=1.0, size=2048)
    mu = np.zeros_like(y)
    sigma = np.ones_like(y)
    pit = pit_zscore(y, mu, sigma)
    p_value = _ks_pvalue_normal(pit)
    assert p_value > 0.05  # FIX: reject only when calibration is poor
