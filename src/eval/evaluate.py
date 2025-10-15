from __future__ import annotations

"""Utilities to evaluate trained Hermite models on their validation slice."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src.config import BINANCE, TRAINING, BinanceAPIConfig, TrainingConfig
from src.data import BinanceDataFetcher, HermiteDataset
from src.features import compute_liquidity_features_series, compute_orderbook_features_series
from src.models import HermiteForecaster


def _load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist")
    checkpoint = torch.load(path, map_location="cpu")
    required_keys = {
        "model_state",
        "feature_mean",
        "feature_std",
        "feature_names",
        "window",
        "horizon",
        "train_time_range",
        "val_time_range",
    }
    missing = required_keys.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")
    return checkpoint
def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    binance_config: BinanceAPIConfig = BINANCE,
    training_config: TrainingConfig = TRAINING,
    csv_output: Path | None = None,
) -> Dict[str, float]:
    """Evaluate a checkpoint and optionally write a per-sample CSV report."""

    checkpoint = _load_checkpoint(checkpoint_path)
    training_config = replace(
        training_config,
        feature_window=int(checkpoint["window"]),
        forecast_horizon=int(checkpoint["horizon"]),
    )

    fetcher = BinanceDataFetcher(binance_config)
    candles = fetcher.get_historical_candles(limit=binance_config.history_limit)
    liquidity_series = compute_liquidity_features_series(candles)
    orderbook_series = compute_orderbook_features_series(candles)
    dataset = HermiteDataset(
        candles,
        liquidity_features=liquidity_series,
        orderbook_features=orderbook_series,
        feature_window=training_config.feature_window,
        forecast_horizon=training_config.forecast_horizon,
        normalise=False,
    )

    feature_mean = checkpoint["feature_mean"].to(torch.float32)
    feature_std = checkpoint["feature_std"].to(torch.float32)
    dataset.features = (dataset.features - feature_mean) / feature_std

    model = HermiteForecaster(input_dim=dataset.features.shape[1], config=training_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_start, val_end = checkpoint["val_time_range"]
    timestamps = dataset.timestamps
    mask = (timestamps >= val_start) & (timestamps <= val_end)
    if not torch.any(mask):
        raise RuntimeError("Validation range from checkpoint not present in dataset")
    val_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    anchor_start = int(timestamps[val_indices[0]].item())
    anchor_end = int(timestamps[val_indices[-1]].item())
    print(
        "Validation anchors -> "
        f"{pd.to_datetime(anchor_start, unit='ms', utc=True).isoformat()} -> "
        f"{pd.to_datetime(anchor_end, unit='ms', utc=True).isoformat()}"
    )

    with torch.no_grad():
        pred_lr, direction_logits = model(dataset.features[val_indices])
    preds = pred_lr.squeeze(1).cpu().numpy()
    direction_logits = direction_logits.squeeze(1).cpu().numpy()

    anchor_prices = dataset.anchor_prices[val_indices].numpy()
    true_log_returns = dataset.targets[val_indices].squeeze(1).numpy()
    pred_prices = anchor_prices * np.exp(preds)
    true_prices = anchor_prices * np.exp(true_log_returns)

    residuals = pred_prices - true_prices
    abs_err = np.abs(residuals)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ape = abs_err / np.maximum(np.abs(true_prices), 1e-6)
    mape = float(ape.mean() * 100.0)
    median_ape = float(np.median(ape) * 100.0)
    avg_last10 = float(abs_err[-10:].mean()) if abs_err.size else float("nan")

    direction_targets = (true_log_returns > 0).astype(int)
    direction_probs = 1.0 / (1.0 + np.exp(-direction_logits))
    direction_pred = (direction_probs >= 0.5).astype(int)
    direction_correct = (direction_pred == direction_targets).astype(int)
    hit_rate = float(direction_correct.mean()) if direction_correct.size else float("nan")

    forecast_frame = pd.DataFrame(
        {
            "anchor_time": pd.to_datetime(dataset.anchor_times[val_indices].numpy(), unit="ms", utc=True),
            "target_time": pd.to_datetime(dataset.target_times[val_indices].numpy(), unit="ms", utc=True),
            "anchor_price": anchor_prices,
            "true_log_return": true_log_returns,
            "pred_log_return": preds,
            "true_price": true_prices,
            "pred_price": pred_prices,
            "abs_error": abs_err,
            "ape_pct": ape * 100.0,
            "direction_prob": direction_probs,
            "direction_pred": direction_pred,
            "direction_hit": direction_correct,
        }
    )

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        forecast_frame.to_csv(csv_output, index=False)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "median_ape": median_ape,
        "avg_abs_err_last_10": avg_last10,
        "direction_hit_rate": hit_rate,
    }

    print(
        "Validation price metrics -> "
        f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.4f}% "
        f"Median APE: {median_ape:.4f}%, Avg abs err last 10: {avg_last10:.6f}, "
        f"Direction hit-rate: {hit_rate:.4f}"
    )

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Hermite NN checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to the saved checkpoint")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional path to write detailed validation predictions as CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_checkpoint(args.checkpoint, csv_output=args.csv_output)


if __name__ == "__main__":
    main()
