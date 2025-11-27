from __future__ import annotations

"""Unit tests for shared utility helpers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.utils.utils import pit_zscore, pt_test, set_seed


def test_set_seed_reproducible() -> None:
    set_seed(123)
    first = np.random.rand(5)
    torch_first = torch.randn(3)
    set_seed(123)
    second = np.random.rand(5)
    torch_second = torch.randn(3)
    assert np.allclose(first, second)
    assert torch.allclose(torch_first, torch_second)


def test_pt_test_extremes() -> None:
    actual = np.array([0, 1, 0, 1, 0, 1])
    predicted = actual.copy()
    p_value = pt_test(actual, predicted)
    assert p_value < 0.05
    rng = np.random.default_rng(0)
    random_pred = rng.integers(0, 2, size=actual.size)
    p_value_random = pt_test(actual, random_pred)
    assert 0.0 <= p_value_random <= 1.0


def test_pit_zscore_shapes() -> None:
    y = np.array([1.0, 2.0, 3.0])
    mu = np.array([0.5, 2.5, 2.5])
    sigma = np.array([0.5, 0.5, 2.0])
    z = pit_zscore(y, mu, sigma)
    assert np.allclose(z, np.array([1.0, -1.0, 0.25]))
