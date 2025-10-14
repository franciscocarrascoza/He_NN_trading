from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from src.config import BINANCE, TRAINING, BinanceAPIConfig, TrainingConfig
from src.data import BinanceDataFetcher
from src.pipeline import HermiteTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hermite NN forecaster for BTCUSDT")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the training checkpoint (overrides config).",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Override the Binance trading pair symbol (e.g. BTCUSDT).")
    parser.add_argument("--interval", type=str, default=None, help="Override the Binance kline interval (e.g. 1h).")
    parser.add_argument("--history-limit", type=int, default=None, help="Number of historical candles to download (<=5000).")
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Number of candles ahead to predict (1-15).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for the optimiser.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--device-preference",
        type=str,
        default=None,
        help="Preferred CUDA device name substring (e.g. RTX 2060).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binance_config: BinanceAPIConfig = BINANCE
    training_config: TrainingConfig = TRAINING

    if any(value is not None for value in (args.symbol, args.interval, args.history_limit)):
        binance_config = replace(
            binance_config,
            symbol=args.symbol or binance_config.symbol,
            interval=args.interval or binance_config.interval,
            history_limit=args.history_limit or binance_config.history_limit,
        )

    if args.forecast_horizon is not None:
        if not 1 <= args.forecast_horizon <= 15:
            raise ValueError("--forecast-horizon must be between 1 and 15 inclusive.")
        training_config = replace(training_config, forecast_horizon=args.forecast_horizon)

    if args.batch_size is not None:
        training_config = replace(training_config, batch_size=args.batch_size)
    if args.learning_rate is not None:
        training_config = replace(training_config, learning_rate=args.learning_rate)
    if args.epochs is not None:
        training_config = replace(training_config, num_epochs=args.epochs)
    if args.device_preference is not None:
        training_config = replace(training_config, device_preference=args.device_preference)

    if args.save is not None:
        training_config = replace(training_config, checkpoint_path=str(args.save))

    fetcher = BinanceDataFetcher(binance_config)
    trainer = HermiteTrainer(fetcher, training_config=training_config)
    artifacts = trainer.train()
    print(f"Training device: {artifacts.device}")
    if artifacts.training_losses:
        print(
            f"Final Huber losses -> Train: {artifacts.training_losses[-1]:.6f}, "
            f"Validation: {artifacts.validation_losses[-1]:.6f}"
        )
    metrics = artifacts.price_metrics
    print(
        "Validation price metrics -> "
        f"MAE: {metrics.mae:.6f}, RMSE: {metrics.rmse:.6f}, MAPE: {metrics.mape:.4f}% "
        f"Median APE: {metrics.median_ape:.4f}%"
    )
    print(
        f"Average absolute error (last 10 val predictions): {metrics.avg_abs_err_last_10:.6f}"
    )
    print(f"Checkpoint stored at: {artifacts.checkpoint_path}")
    next_price = trainer.predict_next_price(artifacts)
    print(f"Predicted next price: {next_price:.2f} USDT")


if __name__ == "__main__":
    main()
