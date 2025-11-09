from __future__ import annotations

"""Utility helpers for reproducibility and diagnostics."""  # FIX: describe deterministic helpers

import math  # FIX: expose scalar math for test statistics
import os  # FIX: environment control for deterministic threading
import random  # FIX: python RNG seeding
import numpy as np  # FIX: rely on numpy for vector utilities
import torch  # FIX: integrate torch seeding alongside numpy


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch for determinism."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_CREATE_SHM", "FALSE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("KMP_FORKJOIN_BARRIER", "plain")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(1)  # pragma: no cover - platform dependent
        except RuntimeError:
            pass
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)  # pragma: no cover - platform dependent
        except RuntimeError:
            pass
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch versions
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pt_test(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return two-sided Pesaran–Timmermann p-value for directional accuracy."""

    actual = np.asarray(actual, dtype=int).ravel()
    predicted = np.asarray(predicted, dtype=int).ravel()
    if actual.size == 0 or actual.shape != predicted.shape:
        return float("nan")
    t = actual.size
    hits = (actual == predicted).astype(float)
    p = hits.mean()
    p1 = actual.mean()
    p2 = predicted.mean()
    expected = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    var = (
        p1 * p2 * (1.0 - p1 * p2)
        + (1.0 - p1) * (1.0 - p2) * (1.0 - (1.0 - p1) * (1.0 - p2))
        - p1 * (1.0 - p1) * p2 * (1.0 - p2)
    )
    var = max(var / max(t, 1), 1e-12)
    z = (p - expected) / math.sqrt(var)
    p_value = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return float(min(1.0, max(0.0, p_value)))


def pit_zscore(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute probability integral transform z-scores."""

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    eps = 1e-6  # FIX: match conformal stability clamp
    safe_sigma = np.clip(sigma, eps, None)
    return (y - mu) / safe_sigma


__all__ = ["pit_zscore", "pt_test", "set_seed"]
