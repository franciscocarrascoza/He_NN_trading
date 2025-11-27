from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURES, FeatureConfig
from src.data import BinanceDataFetcher


def _aggregate_levels(levels: Iterable[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    prices, qtys = zip(*levels) if levels else ([], [])
    return np.asarray(prices, dtype=float), np.asarray(qtys, dtype=float)


def compute_orderbook_features(
    fetcher: BinanceDataFetcher,
    *,
    feature_config: FeatureConfig = FEATURES,
) -> Dict[str, float]:
    """Summarise the futures order book into a compact feature dictionary."""

    snapshot = fetcher.get_order_book(depth=feature_config.order_book_depth)
    bids_prices, bids_qtys = _aggregate_levels(snapshot.get("bids", []))
    asks_prices, asks_qtys = _aggregate_levels(snapshot.get("asks", []))

    best_bid = float(bids_prices[0]) if bids_prices.size else np.nan
    best_ask = float(asks_prices[0]) if asks_prices.size else np.nan
    mid_price = float((best_bid + best_ask) / 2) if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan

    bid_volume = float(bids_qtys.sum())
    ask_volume = float(asks_qtys.sum())
    total_volume = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / (total_volume + 1e-12)

    weighted_bid_price = float((bids_prices * bids_qtys).sum() / (bid_volume + 1e-12)) if bid_volume else np.nan
    weighted_ask_price = float((asks_prices * asks_qtys).sum() / (ask_volume + 1e-12)) if ask_volume else np.nan

    spread = float(best_ask - best_bid) if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan

    features = {
        "orderbook_mid_price": mid_price,
        "orderbook_bid_volume": bid_volume,
        "orderbook_ask_volume": ask_volume,
        "orderbook_imbalance": imbalance,
        "orderbook_weighted_bid_price": weighted_bid_price,
        "orderbook_weighted_ask_price": weighted_ask_price,
        "orderbook_spread": spread,
    }
    return {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in features.items()}


def compute_orderbook_features_series(
    candles: pd.DataFrame,
    *,
    feature_config: FeatureConfig = FEATURES,
) -> pd.DataFrame:
    """Compute order book-style features aligned with each candle.

    These proxies are derived from OHLCV statistics and remain causal.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(candles.columns)
    if missing:
        raise KeyError(f"Candles missing required columns for order book features: {sorted(missing)}")

    frame = candles.reset_index(drop=True).copy()
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float)
    quote_volume = frame.get("quote_asset_volume", close * volume).astype(float)
    trades = frame.get("number_of_trades", pd.Series(0.0, index=frame.index)).astype(float).replace(0.0, np.nan)

    spread = high - low
    mid_price = (high + low) / 2.0
    close_position = (close - low) / (spread + 1e-6)
    price_change = close - open_
    imbalance = price_change / (spread + 1e-6)
    avg_trade_size = quote_volume / (trades + 1e-6)

    def _rolling(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    features = pd.DataFrame(
        {
            "ob_mid_price": mid_price,
            "ob_spread": spread,
            "ob_range_pct": spread / (close.abs() + 1e-6),
            "ob_close_position": close_position,
            "ob_close_position_mean_6": _rolling(close_position, 6),
            "ob_close_position_mean_12": _rolling(close_position, 12),
            "ob_price_imbalance": imbalance.clip(-1.0, 1.0),
            "ob_price_imbalance_mean_6": _rolling(imbalance, 6).clip(-1.0, 1.0),
            "ob_avg_trade_size": avg_trade_size.fillna(0.0),
            "ob_volume_weighted_price": (quote_volume / (volume + 1e-6)),
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.astype(np.float32)


__all__ = ["compute_orderbook_features", "compute_orderbook_features_series"]
