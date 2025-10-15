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
import torch
from torch.utils.data import Dataset

from src.features.causal import compute_causal_features


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

    Each row contains the flattened OHLCV feature window and, optionally,
    additional pre-computed features supplied by the caller. No normalisation is
    applied in this class so that training pipelines can compute statistics from
    the training slice exclusively.
    """

    base_feature_names: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        candles: pd.DataFrame,
        liquidity_features: Dict[str, float] | None = None,
        orderbook_features: Dict[str, float] | None = None,
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
        close_time_series = candles["close_time"]
        if pd.api.types.is_datetime64_any_dtype(close_time_series):
            close_times = (close_time_series.astype("int64") // 1_000_000).to_numpy(dtype=np.int64)
        else:
            close_times = close_time_series.to_numpy(dtype=np.int64)

        liquidity_array: np.ndarray | None = None
        orderbook_array: np.ndarray | None = None
        self.liquidity_feature_names: List[str] = []
        self.orderbook_feature_names: List[str] = []

        if isinstance(liquidity_features, pd.DataFrame):
            liquidity_frame = liquidity_features.reset_index(drop=True).astype(np.float32)
            if len(liquidity_frame) != len(candles):
                raise ValueError("Liquidity feature series must align with candle length.")
            self.liquidity_feature_names = list(liquidity_frame.columns)
            liquidity_array = np.nan_to_num(liquidity_frame.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        elif liquidity_features:
            liquidity_vector = np.array(list(liquidity_features.values()), dtype=np.float32)
            liquidity_array = np.tile(liquidity_vector, (len(candles), 1))
            self.liquidity_feature_names = list(liquidity_features.keys())

        if isinstance(orderbook_features, pd.DataFrame):
            orderbook_frame = orderbook_features.reset_index(drop=True).astype(np.float32)
            if len(orderbook_frame) != len(candles):
                raise ValueError("Order book feature series must align with candle length.")
            self.orderbook_feature_names = list(orderbook_frame.columns)
            orderbook_array = np.nan_to_num(orderbook_frame.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        elif orderbook_features:
            orderbook_vector = np.array(list(orderbook_features.values()), dtype=np.float32)
            orderbook_array = np.tile(orderbook_vector, (len(candles), 1))
            self.orderbook_feature_names = list(orderbook_features.keys())

        extra_features = np.empty(0, dtype=np.float32)
        self.extra_feature_names: List[str] = []
        if liquidity_array is None and isinstance(liquidity_features, dict) and liquidity_features:
            extra_features = np.concatenate([extra_features, np.array(list(liquidity_features.values()), dtype=np.float32)])
            self.extra_feature_names.extend(liquidity_features.keys())
        if orderbook_array is None and isinstance(orderbook_features, dict) and orderbook_features:
            extra_features = np.concatenate([extra_features, np.array(list(orderbook_features.values()), dtype=np.float32)])
            self.extra_feature_names.extend(orderbook_features.keys())
        extra_features = np.nan_to_num(extra_features, nan=0.0, posinf=0.0, neginf=0.0)
        self.extra_features = (
            torch.from_numpy(extra_features.copy()) if extra_features.size else torch.zeros(0, dtype=torch.float32)
        )

        causal_df = compute_causal_features(candles[list(self.base_feature_names)])
        causal_values = np.nan_to_num(causal_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.causal_feature_names = list(causal_df.columns)

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

            components: List[np.ndarray] = [window.flatten().astype(np.float32)]
            if self.causal_feature_names:
                components.append(causal_values[anchor_idx])
            if liquidity_array is not None:
                components.append(liquidity_array[anchor_idx])
            if orderbook_array is not None:
                components.append(orderbook_array[anchor_idx])
            if extra_features.size:
                components.append(extra_features)
            feature_vector = np.concatenate(components, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            raw_target = np.log(future_close / current_close)
            target = float(np.nan_to_num(raw_target, nan=0.0, posinf=0.0, neginf=0.0))

            feature_vectors.append(feature_vector)
            targets.append(target)
            anchor_prices.append(float(current_close))
            anchor_times.append(int(close_times[anchor_idx]))
            target_times.append(int(close_times[future_idx]))

        features_array = np.nan_to_num(np.vstack(feature_vectors), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        targets_array = np.nan_to_num(np.array(targets, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        self.features = torch.from_numpy(features_array)
        self.targets = torch.from_numpy(targets_array)[:, None]
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

        self.feature_names = (
            _build_window_feature_names(self.base_feature_names, window=feature_window)
            + self.causal_feature_names
            + self.liquidity_feature_names
            + self.orderbook_feature_names
            + self.extra_feature_names
        )

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


__all__ = ["HermiteDataset", "DatasetItem"]
