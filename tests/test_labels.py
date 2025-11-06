from __future__ import annotations

import numpy as np
import pytest

from src.data.labels import make_labels


def test_make_labels_basic_horizon() -> None:
    prices = np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float32)
    y_reg, y_bin = make_labels(prices, horizon=1)
    expected_reg = np.log(np.array([1.01, 99.0 / 101.0, 102.0 / 99.0]))
    expected_bin = (expected_reg > 0.0).astype(np.float32)
    np.testing.assert_allclose(y_reg, expected_reg.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(y_bin, expected_bin)


def test_make_labels_requires_enough_prices() -> None:
    with pytest.raises(ValueError):
        make_labels(np.array([1.0], dtype=np.float32), horizon=2)
