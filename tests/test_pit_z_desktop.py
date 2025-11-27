"""Unit tests for PIT z-score computation per spec."""  # FIX: test module docstring per spec

import numpy as np  # FIX: numpy for array operations
import pytest  # FIX: pytest framework
from scipy import stats  # FIX: scipy stats for KS test

from src.utils.utils import pit_zscore  # FIX: import PIT z-score utility


def test_pit_z_normal_distribution():  # FIX: test PIT z for Gaussian data per spec
    """Test that PIT z-scores from mu/sigma have near-normal distribution for Gaussian synthetic data."""  # FIX: test docstring per spec
    np.random.seed(42)  # FIX: set seed for reproducibility per spec
    n = 5000  # FIX: large sample size for KS test

    # FIX: Generate synthetic Gaussian data
    mu = np.zeros(n)  # FIX: zero mean predictions
    sigma = np.ones(n)  # FIX: unit variance predictions
    y = np.random.randn(n)  # FIX: Gaussian targets

    # FIX: Compute PIT z-scores
    pit_z = pit_zscore(y, mu, sigma)  # FIX: PIT z-scores per spec

    # FIX: Perform KS test against standard normal
    ks_stat, ks_p = stats.kstest(pit_z, 'norm')  # FIX: KS test vs N(0,1)

    # FIX: Assert KS p-value > 0.05 (fail to reject normality)
    assert ks_p > 0.05, (
        f"PIT z-scores should be approximately normal for Gaussian data: KS p-value={ks_p:.4f}"
    )  # FIX: assert KS p > 0.05 per spec


def test_pit_z_shape():  # FIX: test PIT z output shape
    """Test that PIT z-scores have correct shape."""  # FIX: test docstring
    n = 100  # FIX: sample size
    y = np.random.randn(n)  # FIX: targets
    mu = np.zeros(n)  # FIX: predictions
    sigma = np.ones(n)  # FIX: sigmas

    # FIX: Compute PIT z-scores
    pit_z = pit_zscore(y, mu, sigma)  # FIX: PIT z-scores

    # FIX: Assert output shape matches input
    assert pit_z.shape == y.shape, "PIT z-scores should have same shape as inputs"  # FIX: assert shape


# FIX: Export test functions
__all__ = ["test_pit_z_normal_distribution", "test_pit_z_shape"]  # FIX: module exports
