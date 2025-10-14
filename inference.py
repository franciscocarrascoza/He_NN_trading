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
from src.models import HermiteForecaster


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
    extra_features: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if len(candles) < window:
        raise ValueError(f"Need at least {window} rows to build inference window")
    candles = candles.sort_values("close_time").reset_index(drop=True)
    window_frame = candles.iloc[-window:]
    if not window_frame["close_time"].is_monotonic_increasing:
        raise ValueError("close_time must be strictly increasing")

    feature_values = []
    for _, row in window_frame.iterrows():
        feature_values.extend(float(row[col]) for col in base_columns)
    feature_vector = torch.tensor(feature_values, dtype=torch.float32)
    feature_vector = torch.cat([feature_vector, extra_features.to(torch.float32)])
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
        raise KeyError("Checkpoint missing 'extra_features'; retrain with the updated pipeline")
    extra_features = extra_features.to(torch.float32)

    base_columns = checkpoint.get("base_feature_names") or list(HermiteDataset.base_feature_names)
    feature_vector, anchor_price = _build_feature_vector(
        candles,
        window=int(checkpoint["window"]),
        base_columns=base_columns,
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
        log_return = model(scaled_features.unsqueeze(0)).item()

    predicted_price = anchor_price * float(np.exp(log_return))

    print(f"Predicted log-return (horizon={training_config.forecast_horizon}): {log_return:.8f}")
    print(f"Anchor price: {anchor_price:.4f}")
    print(f"Predicted future price: {predicted_price:.4f}")


if __name__ == "__main__":
    main()
