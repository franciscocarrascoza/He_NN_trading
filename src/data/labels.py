from __future__ import annotations

from typing import Tuple

import numpy as np


def make_labels(prices: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct regression and binary classification targets for a given forecast horizon.

    Parameters
    ----------
    prices:
        Array of anchor prices ordered chronologically.
    horizon:
        Positive integer horizon (number of steps ahead).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing log returns (shape: n - horizon) and binary labels of the same shape.
    """
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if prices.ndim != 1:
        raise ValueError("Prices array must be 1-dimensional.")
    if prices.size <= horizon:
        raise ValueError("Not enough prices to compute labels at the requested horizon.")

    current = prices[:-horizon].astype(np.float64, copy=False)
    future = prices[horizon:].astype(np.float64, copy=False)

    ratio = future / np.clip(current, 1e-12, None)
    y_reg = np.log(np.clip(ratio, 1e-12, None)).astype(np.float32)
    y_bin = (y_reg > 0.0).astype(np.float32)

    if np.any(np.isnan(y_reg)):
        raise ValueError("NaNs detected in regression targets.")

    return y_reg, y_bin


__all__ = ["make_labels"]
