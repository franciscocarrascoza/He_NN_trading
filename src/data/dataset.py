from __future__ import annotations

"""Dataset utilities for the Hermite neural network forecaster.

The dataset is intentionally free from any normalisation logic. Features are
materialised in chronological order, with each sample `i` representing the
state at anchor timestamp :math:`t_i` (the close price used as the anchor) and
the log-return target between :math:`t_i` and :math:`t_i + H` where `H` is the
forecast horizon.  All temporal indexing is forward looking only; no feature is
allowed to peek beyond its corresponding anchor timestamp.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_float_dtype,
    is_integer_dtype,
)
import torch
from torch.utils.data import Dataset


def _build_window_feature_names(
    base_columns: Sequence[str], *, window: int
) -> List[str]:
    """Return feature names for an ordered, causal rolling window.

    The most recent observation is denoted with ``t`` (anchor candle) and
    earlier observations are expressed as ``t-k``.
    """

    names: List[str] = []
    for offset in range(window):
        step = window - offset - 1
        suffix = "t" if step == 0 else f"t-{step}"
        for column in base_columns:
            names.append(f"{column}_{suffix}")
    return names


@dataclass
class DatasetItem:
    """Single item describing aligned tensors for documentation/testing."""

    features: torch.Tensor
    target: torch.Tensor
    anchor_price: torch.Tensor
    anchor_time: torch.Tensor
    target_time: torch.Tensor


class HermiteDataset(Dataset):
    """Windowed, causal dataset feeding the Hermite neural network.

    Each row contains the flattened OHLCV feature window followed by the
    pre-computed liquidity and order book summary features. No normalisation is
    applied in this class so that training pipelines can compute statistics from
    the training slice exclusively.
    """

    base_feature_names: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        candles: pd.DataFrame,
        liquidity_features: Dict[str, float],
        orderbook_features: Dict[str, float],
        *,
        feature_window: int,
        forecast_horizon: int,
        normalise: bool = False,
    ) -> None:
        super().__init__()
        if feature_window <= 0:
            raise ValueError("feature_window must be positive")
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if "close_time" not in candles:
            raise KeyError("candles must include 'close_time' column")

        candles = candles.reset_index(drop=True).copy()
        if not candles["close_time"].is_monotonic_increasing:
            raise ValueError("candles must be sorted by close_time in ascending order")

        values = candles[list(self.base_feature_names)].to_numpy(dtype=np.float32)

        close_times = _coerce_close_times(candles["close_time"])

        liquidity_vector = np.array(list(liquidity_features.values()), dtype=np.float32)
        orderbook_vector = np.array(list(orderbook_features.values()), dtype=np.float32)
        extra_features = np.concatenate([liquidity_vector, orderbook_vector])
        self.extra_feature_names = list(liquidity_features.keys()) + list(orderbook_features.keys())
        self.extra_features = torch.from_numpy(extra_features.copy())

        feature_vectors: List[np.ndarray] = []
        targets: List[float] = []
        anchor_prices: List[float] = []
        anchor_times: List[int] = []
        target_times: List[int] = []

        total_length = len(candles)
        max_anchor_index = total_length - forecast_horizon - 1
        if max_anchor_index < feature_window - 1:
            raise ValueError("Not enough candles to build at least one sample")

        for anchor_idx in range(feature_window - 1, max_anchor_index + 1):
            window_start = anchor_idx - feature_window + 1
            window_end = anchor_idx + 1  # exclusive in numpy slicing
            window = values[window_start:window_end]
            assert window.shape == (feature_window, len(self.base_feature_names)), (
                "Window shape mismatch; check feature_window bounds.")

            future_idx = anchor_idx + forecast_horizon
            current_close = values[anchor_idx, self.base_feature_names.index("close")]
            future_close = values[future_idx, self.base_feature_names.index("close")]

            feature_vector = np.concatenate([window.flatten(), extra_features], dtype=np.float32)
            target = float(np.log(future_close / current_close))

            feature_vectors.append(feature_vector)
            targets.append(target)
            anchor_prices.append(float(current_close))
            anchor_times.append(int(close_times[anchor_idx]))
            target_times.append(int(close_times[future_idx]))

        self.features = torch.from_numpy(np.vstack(feature_vectors).astype(np.float32))
        self.targets = torch.from_numpy(np.array(targets, dtype=np.float32))[:, None]
        self.anchor_prices = torch.from_numpy(np.array(anchor_prices, dtype=np.float32))
        self.anchor_times = torch.from_numpy(np.array(anchor_times, dtype=np.int64))
        self.target_times = torch.from_numpy(np.array(target_times, dtype=np.int64))
        self.timestamps = self.anchor_times  # alias for training utilities

        assert len(self.features) == len(self.targets) == len(self.anchor_prices) == len(
            self.anchor_times
        ), "Dataset components must align in length"
        assert torch.all(self.target_times > self.anchor_times), "Targets must be strictly in the future"

        self.feature_window = feature_window
        self.forecast_horizon = forecast_horizon
        self.candles = candles
        self.normalised = normalise
        feature_dim = self.features.shape[1]
        self.feature_mean = torch.zeros(feature_dim)
        self.feature_std = torch.ones(feature_dim)

        if normalise:
            raise ValueError(
                "HermiteDataset no longer supports inline normalisation. Set normalise=False "
                "and apply scaling in the training pipeline."
            )

        self.feature_names = _build_window_feature_names(
            self.base_feature_names, window=feature_window
        ) + self.extra_feature_names

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    def item(self, idx: int) -> DatasetItem:
        """Return a rich representation for debugging/testing."""

        return DatasetItem(
            features=self.features[idx],
            target=self.targets[idx],
            anchor_price=self.anchor_prices[idx],
            anchor_time=self.anchor_times[idx],
            target_time=self.target_times[idx],
        )

def _coerce_close_times(series: pd.Series) -> np.ndarray:
    """Return close timestamps as integer milliseconds since epoch.

    Binance candles arrive either as timezone-aware ``datetime64`` (nanosecond
    precision) or as raw integer epochs.  Historically we relied on the numpy
    ``issubdtype`` check which fails for ``DatetimeTZ`` dtypes, leaving the
    timestamps in nanoseconds.  Downstream code then interpreted those values as
    milliseconds and triggered ``pandas`` overflow errors when formatting the
    chronological split.  This helper enforces a single conversion path that
    normalises every supported representation to UTC milliseconds.
    """

    if is_datetime64_any_dtype(series) or is_datetime64tz_dtype(series):
        normalised = pd.to_datetime(series, utc=True, errors="raise")
        values_ns = normalised.view("int64")
    elif is_integer_dtype(series) or is_float_dtype(series):
        numeric = pd.to_numeric(series, errors="raise").astype(np.int64)
        abs_max = int(np.max(np.abs(numeric))) if numeric.size else 0
        if abs_max >= 5_000_000_000_000_000:
            unit = "ns"
        elif abs_max <= 5_000_000_000:
            unit = "s"
        else:
            unit = "ms"
        normalised = pd.to_datetime(numeric, unit=unit, utc=True, errors="raise")
        values_ns = normalised.view("int64")
    else:
        parsed = pd.to_datetime(series, utc=True, errors="raise")
        values_ns = parsed.view("int64")

    return (values_ns // 1_000_000).astype(np.int64)


__all__ = ["HermiteDataset", "DatasetItem"]
