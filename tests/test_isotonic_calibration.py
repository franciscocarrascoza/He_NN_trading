"""Test isotonic calibration improves ECE on toy miscalibrated probabilities."""  # FIX: add test for H3 isotonic per spec

from __future__ import annotations

import numpy as np

from src.pipeline.training import _fit_sklearn_isotonic, _apply_sklearn_isotonic
from src.eval.diagnostics import probability_calibration_metrics


def test_isotonic_improves_ece() -> None:
    """Toy example where p_up_raw monotonic but miscalibrated; isotonic fit reduces Brier and ECE. Assert ECE reduced."""  # FIX: test per spec
    rng = np.random.default_rng(42)  # FIX: deterministic RNG
    n = 500  # FIX: sample size

    true_probs = rng.uniform(0.2, 0.8, size=n)  # FIX: latent true probabilities
    p_raw = np.clip(true_probs * 0.5 + 0.25, 0.01, 0.99)  # FIX: systematically miscalibrated (compressed)
    labels = (rng.uniform(size=n) < true_probs).astype(int)  # FIX: binary labels from true probs

    cal_n = 300  # FIX: calibration set size
    p_raw_cal, p_raw_val = p_raw[:cal_n], p_raw[cal_n:]  # FIX: split calibration and validation
    labels_cal, labels_val = labels[:cal_n], labels[cal_n:]  # FIX: split labels

    calibrator = _fit_sklearn_isotonic(p_raw_cal, labels_cal)  # FIX: fit isotonic on calibration per spec
    p_iso_val = _apply_sklearn_isotonic(calibrator, p_raw_val)  # FIX: apply to validation

    metrics_raw = probability_calibration_metrics(labels_val, p_raw_val, n_bins=10)  # FIX: raw metrics
    metrics_iso = probability_calibration_metrics(labels_val, p_iso_val, n_bins=10)  # FIX: isotonic-calibrated metrics

    assert metrics_iso.ece < metrics_raw.ece  # FIX: ECE must reduce after isotonic calibration per spec
    assert metrics_iso.brier <= metrics_raw.brier + 0.01  # FIX: Brier should not increase significantly (allow small numerical noise)


def test_isotonic_preserves_rank_order() -> None:
    """Isotonic calibration preserves rank order of probabilities."""  # FIX: additional coverage for monotonicity
    rng = np.random.default_rng(123)  # FIX: deterministic seed
    n = 100  # FIX: sample size
    p_raw = rng.uniform(0.1, 0.9, size=n)  # FIX: raw probabilities
    labels = rng.integers(0, 2, size=n)  # FIX: binary labels

    calibrator = _fit_sklearn_isotonic(p_raw, labels)  # FIX: fit calibrator
    p_iso = _apply_sklearn_isotonic(calibrator, p_raw)  # FIX: apply calibrator

    raw_ranks = np.argsort(np.argsort(p_raw))  # FIX: rank order of raw probs
    iso_ranks = np.argsort(np.argsort(p_iso))  # FIX: rank order of calibrated probs

    assert np.array_equal(raw_ranks, iso_ranks)  # FIX: isotonic preserves rank order by construction
