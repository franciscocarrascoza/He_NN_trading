from __future__ import annotations

"""Causal feature engineering utilities built from historical OHLCV data."""

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _safe_fill(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Replace NaNs/Infs with zeros."""

    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def compute_causal_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute causal, per-bar features using only information available at time ``t``.

    Parameters
    ----------
    candles:
        Candle dataframe with columns ``open``, ``high``, ``low``, ``close``, ``volume``.

    Returns
    -------
    pd.DataFrame
        Dataframe aligned with ``candles`` containing engineered features.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(candles.columns)
    if missing:
        raise KeyError(f"Candles missing required columns: {sorted(missing)}")

    df = candles.reset_index(drop=True).astype(float)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    log_ret_1 = np.log(close / close.shift(1))

    features: Dict[str, pd.Series] = {}

    # Log returns over varying horizons
    features["log_ret_1"] = log_ret_1
    for horizon in (3, 6, 12):
        features[f"log_ret_{horizon}"] = np.log(close / close.shift(horizon))

    # Rolling volatility of log returns
    for window in (6, 12, 24):
        features[f"volatility_{window}"] = log_ret_1.rolling(window, min_periods=1).std()

    # Average true range
    features["atr_14"] = _atr(high, low, close, period=14)

    # Trend indicators
    ema_fast = _ema(close, span=6)
    ema_slow = _ema(close, span=12)
    features["ema_6"] = ema_fast
    features["ema_12"] = ema_slow
    features["ema_diff_6_12"] = ema_fast - ema_slow
    rolling_std_12 = close.rolling(window=12, min_periods=1).std()
    features["price_zscore_ema12"] = (close - ema_slow) / (rolling_std_12 + 1e-6)

    # Momentum indicators
    features["rsi_14"] = _rsi(close, period=14)
    ema_12 = _ema(close, span=12)
    ema_26 = _ema(close, span=26)
    macd = ema_12 - ema_26
    macd_signal = _ema(macd, span=9)
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd - macd_signal

    # Volume features
    for window in (12, 24):
        mean = volume.rolling(window, min_periods=1).mean()
        std = volume.rolling(window, min_periods=1).std()
        features[f"volume_zscore_{window}"] = (volume - mean) / (std + 1e-6)

    # Realised volatility
    realised_var = (log_ret_1.pow(2)).rolling(window=12, min_periods=1).sum()
    features["realised_vol_12"] = np.sqrt(realised_var)

    feature_df = pd.DataFrame(features)
    feature_df = _safe_fill(feature_df)
    return feature_df.astype(np.float32)


__all__ = ["compute_causal_features"]

