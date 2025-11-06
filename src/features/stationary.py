from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import DataConfig


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_autocorr(series: pd.Series, window: int) -> pd.Series:
    if window < 2:
        raise ValueError("autocorr_window must be >= 2")

    def _autocorr(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        centred = values - values.mean()
        denom = float(np.dot(centred, centred))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(centred[:-1], centred[1:]) / denom)

    return series.rolling(window=window, min_periods=2).apply(_autocorr, raw=True).fillna(0.0)


def _resolve_close_times(series: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    numeric = series.astype("int64")
    # Heuristic: if values look like nanoseconds, downscale to milliseconds first.
    max_abs = np.abs(numeric.to_numpy()).max(initial=0)
    if max_abs > 10**14:
        numeric = numeric // 1_000_000
    return pd.to_datetime(numeric, unit="ms", utc=True)


def compute_stationary_features(candles: pd.DataFrame, *, config: DataConfig) -> pd.DataFrame:
    required_cols = {"open", "high", "low", "close", "volume", "close_time"}
    missing = required_cols.difference(candles.columns)
    if missing:
        raise KeyError(f"Candles missing required columns: {sorted(missing)}")

    df = candles.reset_index(drop=True).copy()
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    prev_close = close.shift(1)
    log_ret_close = np.log(close / prev_close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    hl_range = _safe_divide(high - low, prev_close)
    oc_gap = _safe_divide(open_ - prev_close, prev_close)

    feature_columns: dict[str, pd.Series] = {
        "log_ret_close": log_ret_close,
        "hl_range": hl_range,
        "oc_gap": oc_gap,
        "volume": volume.fillna(0.0),
    }

    for window in config.momentum_windows:
        series = log_ret_close.rolling(window=window, min_periods=1).mean().fillna(0.0)
        feature_columns[f"log_ret_mean_{window}"] = series

    for window in config.volatility_windows:
        series = log_ret_close.rolling(window=window, min_periods=1).std().fillna(0.0)
        feature_columns[f"log_ret_std_{window}"] = series.replace([np.inf, -np.inf], 0.0)

    feature_columns["log_ret_autocorr_1"] = _rolling_autocorr(log_ret_close, config.autocorr_window)

    rolling_max = close.rolling(window=config.drawdown_window, min_periods=1).max()
    feature_columns["recent_drawdown"] = _safe_divide(close, rolling_max) - 1.0

    close_times = _resolve_close_times(df["close_time"])
    if isinstance(close_times, pd.DatetimeIndex):
        dow = close_times.dayofweek
    else:
        dow = close_times.dt.dayofweek
    dow_dummy = pd.get_dummies(dow, prefix="dow", dtype=float)
    expected_cols = [f"dow_{idx}" for idx in range(7)]
    for col in expected_cols:
        if col not in dow_dummy:
            dow_dummy[col] = 0.0
    dow_dummy = dow_dummy[expected_cols]

    feature_frame = pd.DataFrame(feature_columns)
    feature_frame = pd.concat([feature_frame, dow_dummy], axis=1)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feature_frame


__all__ = ["compute_stationary_features"]
