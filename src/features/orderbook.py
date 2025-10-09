from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

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

    return {
        "orderbook_mid_price": mid_price,
        "orderbook_bid_volume": bid_volume,
        "orderbook_ask_volume": ask_volume,
        "orderbook_imbalance": imbalance,
        "orderbook_weighted_bid_price": weighted_bid_price,
        "orderbook_weighted_ask_price": weighted_ask_price,
        "orderbook_spread": spread,
    }


__all__ = ["compute_orderbook_features"]
