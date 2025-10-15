from __future__ import annotations

"""Standalone inference script for Hermite NN checkpoints."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from src.config import TRAINING, TrainingConfig
from src.data.dataset import HermiteDataset
from src.features import (
    compute_causal_features,
    compute_liquidity_features_series,
    compute_orderbook_features_series,
)
from src.models import HermiteForecaster
from src.features.causal import compute_causal_features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a Hermite NN checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the training checkpoint")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help=(
            "Path to a CSV containing the most recent candles with columns "
            "open, high, low, close, volume, close_time"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional override for the forecast horizon (defaults to checkpoint value)",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    required_keys = {"model_state", "feature_mean", "feature_std", "feature_names", "window"}
    missing = required_keys.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint {path} missing keys: {sorted(missing)}")
    return checkpoint


def _build_feature_vector(
    candles: pd.DataFrame,
    *,
    window: int,
    base_columns: Sequence[str],
    causal_features: pd.DataFrame,
    causal_feature_names: Sequence[str],
    liquidity_features: pd.DataFrame,
    liquidity_feature_names: Sequence[str],
    orderbook_features: pd.DataFrame,
    orderbook_feature_names: Sequence[str],
    extra_features: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if len(candles) < window:
        raise ValueError(f"Need at least {window} rows to build inference window")
    candles = candles.sort_values("close_time").reset_index(drop=True)
    window_frame = candles.iloc[-window:]
    if not window_frame["close_time"].is_monotonic_increasing:
        raise ValueError("close_time must be strictly increasing")

    base_values = window_frame[base_columns].to_numpy(dtype=np.float32).reshape(-1)

    def _reindex_latest(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
        if not names:
            return np.empty(0, dtype=np.float32)
        series = frame.iloc[-1].reindex(names, fill_value=0.0)
        return np.nan_to_num(series.to_numpy(dtype=np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)

    causal_vector = _reindex_latest(causal_features, causal_feature_names)
    liquidity_vector = _reindex_latest(liquidity_features, liquidity_feature_names)
    orderbook_vector = _reindex_latest(orderbook_features, orderbook_feature_names)
    extra_vector = (
        extra_features.to(torch.float32).cpu().numpy() if extra_features.numel() else np.empty(0, dtype=np.float32)
    )

    components = [base_values]
    if causal_vector.size:
        components.append(causal_vector)
    if liquidity_vector.size:
        components.append(liquidity_vector)
    if orderbook_vector.size:
        components.append(orderbook_vector)
    if extra_vector.size:
        components.append(extra_vector)

    feature_array = np.concatenate(components, dtype=np.float32)
    feature_vector = torch.from_numpy(feature_array)
    anchor_price = float(window_frame.iloc[-1]["close"])
    return feature_vector, anchor_price


def main() -> None:
    args = _parse_args()
    checkpoint = _load_checkpoint(args.ckpt)
    training_data = checkpoint.get("training_config", {})

    training_config: TrainingConfig
    if training_data:
        training_config = replace(TRAINING, **training_data)
    else:
        training_config = replace(
            TRAINING,
            feature_window=int(checkpoint["window"]),
            forecast_horizon=int(checkpoint.get("horizon", TRAINING.forecast_horizon)),
        )

    if args.horizon is not None:
        training_config = replace(training_config, forecast_horizon=args.horizon)

    input_dim = int(checkpoint["feature_mean"].shape[0])
    model = HermiteForecaster(input_dim=input_dim, config=training_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    candles = pd.read_csv(args.csv)
    for column in ("open", "high", "low", "close", "volume", "close_time"):
        if column not in candles:
            raise KeyError(f"CSV missing required column '{column}'")

    extra_features = checkpoint.get("extra_features")
    if extra_features is None:
        extra_features = torch.zeros(0, dtype=torch.float32)
    else:
        extra_features = extra_features.to(torch.float32)

    base_columns = checkpoint.get("base_feature_names") or list(HermiteDataset.base_feature_names)
    causal_features_df = compute_causal_features(candles[base_columns])
    causal_feature_names = checkpoint.get("causal_feature_names") or list(causal_features_df.columns)
    if isinstance(causal_feature_names, tuple):
        causal_feature_names = list(causal_feature_names)
    liquidity_features_df = compute_liquidity_features_series(candles)
    liquidity_feature_names = checkpoint.get("liquidity_feature_names") or list(liquidity_features_df.columns)
    if isinstance(liquidity_feature_names, tuple):
        liquidity_feature_names = list(liquidity_feature_names)
    orderbook_features_df = compute_orderbook_features_series(candles)
    orderbook_feature_names = checkpoint.get("orderbook_feature_names") or list(orderbook_features_df.columns)
    if isinstance(orderbook_feature_names, tuple):
        orderbook_feature_names = list(orderbook_feature_names)
    feature_vector, anchor_price = _build_feature_vector(
        candles,
        window=int(checkpoint["window"]),
        base_columns=base_columns,
        causal_features=causal_features_df,
        causal_feature_names=causal_feature_names,
        liquidity_features=liquidity_features_df,
        liquidity_feature_names=liquidity_feature_names,
        orderbook_features=orderbook_features_df,
        orderbook_feature_names=orderbook_feature_names,
        extra_features=extra_features,
    )

    feature_mean = checkpoint["feature_mean"].to(torch.float32)
    feature_std = checkpoint["feature_std"].to(torch.float32)
    if feature_vector.shape[0] != feature_mean.shape[0]:
        raise ValueError(
            "Feature vector dimension does not match the saved scaler statistics"
        )
    scaled_features = (feature_vector - feature_mean) / feature_std

    with torch.no_grad():
        log_return_tensor, direction_logits = model(scaled_features.unsqueeze(0))
        log_return = log_return_tensor.item()
        direction_prob = torch.sigmoid(direction_logits).item()

    predicted_price = anchor_price * float(np.exp(log_return))

    print(f"Predicted log-return (horizon={training_config.forecast_horizon}): {log_return:.8f}")
    print(f"Anchor price: {anchor_price:.4f}")
    print(f"Predicted future price: {predicted_price:.4f}")
    print(f"Probability price increases: {direction_prob:.4%}")


if __name__ == "__main__":
    main()
