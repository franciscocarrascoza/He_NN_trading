from __future__ import annotations

"""Dataset utilities for stationary, leak-safe Hermite NN training."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import DATA, FEATURES, DataConfig, FeatureConfig
from src.data.labels import make_labels
from src.features import compute_stationary_features


def _build_window_feature_names(base_columns: Sequence[str], *, window: int) -> List[str]:
    names: List[str] = []
    for offset in range(window):
        step = window - offset - 1
        suffix = "t" if step == 0 else f"t-{step}"
        for column in base_columns:
            names.append(f"{column}_{suffix}")
    return names


@dataclass
class DatasetItem:
    features: torch.Tensor
    target: torch.Tensor
    direction_label: torch.Tensor
    anchor_price: torch.Tensor
    anchor_time: torch.Tensor
    target_time: torch.Tensor


class HermiteDataset(Dataset):
    """Windowed dataset built from stationary features."""

    window_feature_columns: Tuple[str, ...] = ("log_ret_close", "hl_range", "oc_gap", "volume")

    def __init__(
        self,
        candles: pd.DataFrame,
        *,
        data_config: DataConfig = DATA,
        feature_config: FeatureConfig = FEATURES,
        liquidity_features: pd.DataFrame | None = None,
        orderbook_features: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        if data_config.feature_window <= 0:
            raise ValueError("feature_window must be positive.")
        if data_config.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if "close_time" not in candles:
            raise KeyError("candles must include 'close_time' column.")

        self.data_config = data_config
        self.feature_config = feature_config

        candles = candles.reset_index(drop=True).copy()
        if not candles["close_time"].is_monotonic_increasing:
            raise ValueError("candles must be sorted by close_time in ascending order.")

        stationary_frame = compute_stationary_features(
            candles,
            config=data_config,
            feature_config=feature_config,
        )
        if len(stationary_frame) != len(candles):
            raise ValueError("Stationary feature frame must align with candle length.")

        use_liquidity = liquidity_features is not None
        use_orderbook = orderbook_features is not None

        if use_liquidity:
            liquidity_features = liquidity_features.reset_index(drop=True).astype(np.float32)
            if len(liquidity_features) != len(candles):
                raise ValueError("Liquidity feature frame must align with candle length.")
        if use_orderbook:
            orderbook_features = orderbook_features.reset_index(drop=True).astype(np.float32)
            if len(orderbook_features) != len(candles):
                raise ValueError("Order book feature frame must align with candle length.")

        window_features = stationary_frame[list(self.window_feature_columns)].astype(np.float32)
        context_columns = [
            column
            for column in stationary_frame.columns
            if column not in self.window_feature_columns
        ]
        context_features = stationary_frame[context_columns].astype(np.float32)

        close_series = candles["close"].astype(np.float32)
        close_time_series = candles["close_time"]
        if pd.api.types.is_datetime64_any_dtype(close_time_series):
            anchor_times_np = (close_time_series.astype("int64") // 1_000_000).to_numpy(dtype=np.int64)
        else:
            anchor_times_np = close_time_series.to_numpy(dtype=np.int64)

        total_length = len(candles)
        feature_window = data_config.feature_window
        forecast_horizon = data_config.forecast_horizon

        max_anchor_index = total_length - forecast_horizon - 1
        if max_anchor_index < feature_window - 1:
            raise ValueError("Not enough candles to build at least one sample.")

        feature_vectors: List[np.ndarray] = []
        targets: List[float] = []
        direction_labels: List[int] = []
        anchor_prices: List[float] = []
        anchor_times: List[int] = []
        target_times: List[int] = []

        liquidity_names = list(liquidity_features.columns) if use_liquidity else []
        orderbook_names = list(orderbook_features.columns) if use_orderbook else []

        window_feature_names = _build_window_feature_names(self.window_feature_columns, window=feature_window)
        self.context_feature_names = context_columns
        self.liquidity_feature_names = liquidity_names
        self.orderbook_feature_names = orderbook_names

        prices_np = close_series.to_numpy(copy=False)
        full_reg, full_bin = make_labels(prices_np, forecast_horizon)
        if np.isnan(full_reg).any():
            raise ValueError("NaNs detected in regression targets.")

        for anchor_idx in range(feature_window - 1, max_anchor_index + 1):
            window_start = anchor_idx - feature_window + 1
            window_end = anchor_idx + 1
            window_slice = window_features.iloc[window_start:window_end].to_numpy(copy=False).flatten()
            context_slice = context_features.iloc[anchor_idx].to_numpy(copy=False)

            components = [window_slice, context_slice]
            if use_liquidity:
                components.append(liquidity_features.iloc[anchor_idx].to_numpy(copy=False))
            if use_orderbook:
                components.append(orderbook_features.iloc[anchor_idx].to_numpy(copy=False))

            feature_vector = np.concatenate(components, dtype=np.float32)

            future_idx = anchor_idx + forecast_horizon
            current_close = float(close_series.iloc[anchor_idx])

            feature_vectors.append(feature_vector)
            targets.append(float(full_reg[anchor_idx]))
            direction_labels.append(int(full_bin[anchor_idx]))
            anchor_prices.append(current_close)
            anchor_times.append(int(anchor_times_np[anchor_idx]))
            target_times.append(int(anchor_times_np[future_idx]))

        features_array = np.vstack(feature_vectors).astype(np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        direction_array = np.array(direction_labels, dtype=np.int64)

        self.features = torch.from_numpy(features_array)
        self.targets = torch.from_numpy(targets_array)[:, None]
        self.direction_labels = torch.from_numpy(direction_array)[:, None]
        self.anchor_prices = torch.from_numpy(np.array(anchor_prices, dtype=np.float32))
        self.anchor_times = torch.from_numpy(np.array(anchor_times, dtype=np.int64))
        self.target_times = torch.from_numpy(np.array(target_times, dtype=np.int64))
        self.timestamps = self.anchor_times

        assert len(self.features) == len(self.targets), "Features and targets must align."
        assert torch.all(self.target_times > self.anchor_times), "Targets must be strictly in the future."
        assert self.direction_labels.shape == self.targets.shape, "Direction label shape must match targets."
        # Confirm classification labels encode the same forecast horizon as regression targets.
        reconstructed_labels = (self.targets.view(-1) > 0.0).to(torch.int64)
        if not torch.equal(reconstructed_labels, self.direction_labels.view(-1)):
            raise ValueError("Direction labels mismatch regression target sign for the given horizon.")

        self.feature_window = feature_window
        self.forecast_horizon = forecast_horizon
        self.candles = candles
        self.stationary_frame = stationary_frame

        feature_dim = self.features.shape[1]
        self.feature_mean = torch.zeros(feature_dim)
        self.feature_std = torch.ones(feature_dim)

        self.window_feature_names = window_feature_names
        # Replace volume suffix with vol_z to reflect z-scored scaling expectation.
        self.window_feature_names = [
            name.replace("volume", "vol_z") for name in self.window_feature_names
        ]

        self.feature_names = (
            self.window_feature_names
            + list(self.context_feature_names)
            + self.liquidity_feature_names
            + self.orderbook_feature_names
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    def item(self, idx: int) -> DatasetItem:
        return DatasetItem(
            features=self.features[idx],
            target=self.targets[idx],
            direction_label=self.direction_labels[idx],
            anchor_price=self.anchor_prices[idx],
            anchor_time=self.anchor_times[idx],
            target_time=self.target_times[idx],
        )


__all__ = ["HermiteDataset", "DatasetItem"]
