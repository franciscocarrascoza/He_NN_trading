"""Unit tests for isotonic regression calibration per spec."""  # FIX: test module docstring per spec

import numpy as np  # FIX: numpy for array operations
import pytest  # FIX: pytest framework

from src.eval.diagnostics import probability_calibration_metrics  # FIX: import calibration metrics

try:  # FIX: try importing sklearn isotonic
    from sklearn.isotonic import IsotonicRegression  # FIX: sklearn isotonic regression
    SKLEARN_AVAILABLE = True  # FIX: sklearn available flag
except ImportError:  # FIX: handle sklearn unavailable
    SKLEARN_AVAILABLE = False  # FIX: sklearn unavailable flag


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")  # FIX: skip test if sklearn unavailable
def test_isotonic_reduces_ece():  # FIX: test isotonic reduces ECE per spec
    """Test that isotonic regression reduces ECE on miscalibrated toy dataset."""  # FIX: test docstring per spec
    np.random.seed(42)  # FIX: set seed for reproducibility per spec
    n = 1000  # FIX: sample size

    # FIX: Generate synthetic labels (binary classification)
    labels = (np.random.rand(n) > 0.5).astype(int)  # FIX: random binary labels

    # FIX: Generate miscalibrated probabilities (monotone but shifted)
    raw_probs = np.clip(labels + np.random.randn(n) * 0.3 + 0.2, 0.01, 0.99)  # FIX: miscalibrated probabilities

    # FIX: Compute ECE before calibration
    metrics_before = probability_calibration_metrics(labels, raw_probs, n_bins=10)  # FIX: raw metrics
    ece_before = metrics_before.ece  # FIX: ECE before calibration

    # FIX: Split into calibration and validation
    n_calib = n // 2  # FIX: calibration size
    calib_probs = raw_probs[:n_calib]  # FIX: calibration probabilities
    calib_labels = labels[:n_calib]  # FIX: calibration labels
    val_probs = raw_probs[n_calib:]  # FIX: validation probabilities
    val_labels = labels[n_calib:]  # FIX: validation labels

    # FIX: Fit isotonic regression calibrator
    iso = IsotonicRegression(out_of_bounds="clip")  # FIX: isotonic with clip mode per spec
    iso.fit(calib_probs, calib_labels)  # FIX: fit on calibration set

    # FIX: Apply isotonic calibration to validation set
    calibrated_probs = iso.predict(val_probs)  # FIX: apply calibration

    # FIX: Compute ECE after calibration
    metrics_after = probability_calibration_metrics(val_labels, calibrated_probs, n_bins=10)  # FIX: calibrated metrics
    ece_after = metrics_after.ece  # FIX: ECE after calibration

    # FIX: Assert ECE reduction
    assert ece_after < ece_before, (
        f"Isotonic calibration should reduce ECE: before={ece_before:.4f}, after={ece_after:.4f}"
    )  # FIX: assert ECE reduction per spec


# FIX: Export test functions
__all__ = ["test_isotonic_reduces_ece"]  # FIX: module exports
