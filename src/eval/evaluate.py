from __future__ import annotations

"""Utilities to evaluate trained Hermite models on their validation slice."""  # FIX: extend evaluation with diagnostics

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src.config import (  # FIX: access reporting config for diagnostic outputs
    BINANCE,
    DATA,
    FEATURES,
    MODEL,
    REPORTING,
    TRAINING,
    BinanceAPIConfig,
    ReportingConfig,
    TrainingConfig,
)
from src.data import BinanceDataFetcher, HermiteDataset
from src.features import compute_liquidity_features_series, compute_orderbook_features_series
from src.eval.diagnostics import mincer_zarnowitz, save_pit_qq_plot, save_sigma_histogram  # FIX: reuse diagnostics helpers
from src.models import HermiteForecaster
from src.reporting.plots import save_mz_scatter  # FIX: leverage reporting plot for MZ scatter
from src.utils.utils import pit_zscore


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
    reporting_config: ReportingConfig = REPORTING,
    csv_output: Path | None = None,
) -> Dict[str, float]:
    """Evaluate a checkpoint and optionally write a per-sample CSV report."""  # FIX: document reporting integration

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
        data_config=replace(DATA, feature_window=training_config.feature_window, forecast_horizon=training_config.forecast_horizon),
        feature_config=FEATURES,
        liquidity_features=liquidity_series,
        orderbook_features=orderbook_series,
    )

    feature_mean = checkpoint["feature_mean"].to(torch.float32)
    feature_std = checkpoint["feature_std"].to(torch.float32)
    dataset.features = (dataset.features - feature_mean) / feature_std

    model = HermiteForecaster(
        input_dim=dataset.features.shape[1],
        model_config=MODEL,
        feature_window=dataset.feature_window,
        window_feature_columns=HermiteDataset.window_feature_columns,
    )
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
        output = model(dataset.features[val_indices])
    preds = output.mu.squeeze(1).cpu().numpy()
    logvar = output.logvar.squeeze(1).cpu().numpy()  # FIX: extract log-variance head output
    variance = np.exp(logvar)  # FIX: convert log-variance to variance domain
    sigma = np.sqrt(np.clip(variance, 1e-6, None))  # FIX: stabilise sigma extraction
    direction_logits = output.logits.squeeze(1).cpu().numpy()
    direction_probs = output.probability(MODEL.prob_source).squeeze(1).cpu().numpy()
    prob_raw = output.p_up_logit.squeeze(1).cpu().numpy()  # FIX: capture raw logistic probabilities
    prob_cdf = output.p_up_cdf.squeeze(1).cpu().numpy()  # FIX: capture Gaussian CDF probabilities

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
    direction_pred = (direction_probs >= 0.5).astype(int)
    direction_correct = (direction_pred == direction_targets).astype(int)
    hit_rate = float(direction_correct.mean()) if direction_correct.size else float("nan")

    conformal_placeholder = np.full_like(preds, np.nan, dtype=float)  # FIX: no conformal calibration in checkpoint eval
    pit_z = pit_zscore(true_log_returns, preds, sigma)  # FIX: compute PIT z-scores for diagnostics
    mz = mincer_zarnowitz(true_log_returns, preds)  # FIX: run Mincer–Zarnowitz regression on evaluation slice

    forecast_frame = pd.DataFrame(
        {
            "anchor_time": pd.to_datetime(dataset.anchor_times[val_indices].numpy(), unit="ms", utc=True),
            "target_time": pd.to_datetime(dataset.target_times[val_indices].numpy(), unit="ms", utc=True),
            "anchor_price": anchor_prices,
            "true_log_return": true_log_returns,
            "pred_log_return": preds,
            "pred_mean": preds,  # FIX: expose predictive mean explicitly for diagnostics
            "true_price": true_prices,
            "pred_price": pred_prices,
            "abs_error": abs_err,
            "ape_pct": ape * 100.0,
            "direction_prob": direction_probs,
            "direction_pred": direction_pred,
            "direction_hit": direction_correct,
            "sigma": sigma,  # FIX: include predictive dispersion
            "pred_sigma": sigma,  # FIX: duplicate sigma column for PIT export naming
            "logit": direction_logits,  # FIX: expose directional logits
            "p_up_raw": prob_raw,  # FIX: raw logistic probabilities
            "p_up_cal": direction_probs,  # FIX: calibrated probabilities placeholder
            "p_up_cdf": prob_cdf,  # FIX: Gaussian CDF probabilities
            "conformal_p": conformal_placeholder,  # FIX: placeholder conformal p-values
            "pit_z": pit_z,  # FIX: PIT z-score diagnostic
        }
    )

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        forecast_frame.to_csv(csv_output, index=False)

    report_dir = Path(reporting_config.output_dir).resolve()  # FIX: ensure diagnostics follow reporting location
    report_dir.mkdir(parents=True, exist_ok=True)  # FIX: allow saving when directory absent
    pit_dir = report_dir / "pit_eval"  # FIX: dedicate PIT exports subdirectory
    pit_dir.mkdir(parents=True, exist_ok=True)
    pit_frame = forecast_frame[[  # FIX: select canonical diagnostic columns
        "target_time",
        "true_log_return",
        "pred_mean",
        "pred_sigma",
        "logit",
        "p_up_raw",
        "p_up_cal",
        "conformal_p",
        "pit_z",
    ]].rename(
        columns={
            "target_time": "timestamp",
            "true_log_return": "y",
            "pred_mean": "mu",
            "pred_sigma": "sigma",
        }
    )  # FIX: align with reporting specification
    pit_path = pit_dir / f"pit_{checkpoint_path.stem}.csv"  # FIX: deterministic PIT output file
    pit_frame.to_csv(pit_path, index=False)

    plots_dir = report_dir / "plots_eval"  # FIX: dedicated evaluation plots folder
    plots_dir.mkdir(parents=True, exist_ok=True)
    sigma_hist_path = plots_dir / f"sigma_hist_{checkpoint_path.stem}.png"  # FIX: sigma histogram path
    save_sigma_histogram(sigma, output_path=sigma_hist_path, title="Predictive σ Histogram (eval)")  # FIX: persist sigma histogram
    pit_plot_path = plots_dir / f"pit_qq_{checkpoint_path.stem}.png"  # FIX: PIT QQ path
    save_pit_qq_plot(pit_z, output_path=pit_plot_path, title="PIT QQ Plot (eval)")  # FIX: persist PIT QQ diagnostic
    mz_plot_path = plots_dir / f"mz_scatter_{checkpoint_path.stem}.png"  # FIX: scatter path for MZ diagnostics
    save_mz_scatter(
        preds,
        true_log_returns,
        intercept=mz.intercept,
        slope=mz.slope,
        output_path=mz_plot_path,
        title="Mincer–Zarnowitz Scatter (eval)",
    )  # FIX: visualise regression fit

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "median_ape": median_ape,
        "avg_abs_err_last_10": avg_last10,
        "direction_hit_rate": hit_rate,
        "mz_intercept": mz.intercept,  # FIX: expose OLS intercept for diagnostics
        "mz_slope": mz.slope,  # FIX: expose OLS slope for diagnostics
        "mz_p_value": mz.p_value,  # FIX: expose joint hypothesis p-value
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
