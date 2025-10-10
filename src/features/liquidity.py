from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.config import FEATURES, FeatureConfig
from src.data import BinanceDataFetcher

MMR = 0.004
LEVERAGE_TIERS = np.array([5, 10, 20, 50, 75, 100], dtype=float)


@dataclass
class LiquiditySnapshot:
    """Container for the processed liquidity map."""

    price: float
    density: pd.DataFrame

    def summary(self, top_k: int = FEATURES.liquidity_top_k) -> Dict[str, float]:
        df = self.density
        top = df.nlargest(top_k, "liq_density")
        weighted_price = (df["price"] * df["liq_density"]).sum() / (df["liq_density"].sum() + 1e-12)
        features: Dict[str, float] = {
            "liq_density_mean": float(df["liq_density"].mean()),
            "liq_density_std": float(df["liq_density"].std()),
            "liq_density_max": float(df["liq_density"].max()),
            "liq_density_min": float(df["liq_density"].min()),
            "liq_density_weighted_price": float(weighted_price),
            "liq_density_price_skew": float((df["price"] - self.price).pow(3).mean()),
        }
        for idx, row in enumerate(top.itertuples(index=False), start=1):
            features[f"liq_top_{idx}_price"] = float(row.price)
            features[f"liq_top_{idx}_density"] = float(row.liq_density)
        return features


def _coinglass_style_liquidity(
    fetcher: BinanceDataFetcher,
    *,
    feature_config: FeatureConfig = FEATURES,
) -> LiquiditySnapshot:
    price = fetcher.get_mark_price()
    open_interest = fetcher.get_open_interest()
    long_ratio_entry = fetcher.get_long_short_ratio(limit=1)[0]
    long_ratio = float(long_ratio_entry.get("longAccount", 0.5))
    short_ratio = max(0.0, min(1.0, 1.0 - long_ratio))
    funding_rate_entry = fetcher.get_funding_rate(limit=1)[0]
    funding_rate = float(funding_rate_entry.get("fundingRate", 0.0))

    candles = fetcher.get_historical_candles(limit=24)
    candles = candles[["close", "volume"]].astype(float)

    price_min = price * (1 - feature_config.liquidity_price_range)
    price_max = price * (1 + feature_config.liquidity_price_range)
    bins_edges = np.linspace(price_min, price_max, feature_config.liquidity_bins + 1)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

    candles["price_bin"] = pd.cut(
        candles["close"],
        bins=bins_edges,
        labels=bin_centers,
        include_lowest=True,
    )
    volume_by_price = (
        candles.groupby("price_bin", observed=True)["volume"].sum().reindex(bin_centers, fill_value=0.0)
    )
    total_volume = volume_by_price.sum()
    volume_weights = volume_by_price / (total_volume + 1e-12)

    leverage_weights = np.array([0.3, 0.3, 0.25, 0.15, 0.1, 0.1], dtype=float)
    if funding_rate > 0.0001:
        leverage_weights[-2:] += 0.1
    leverage_weights /= leverage_weights.sum()

    density = np.zeros_like(bin_centers)
    for weight, leverage in zip(leverage_weights, LEVERAGE_TIERS):
        long_liq_price = price * (1 - (1.0 / leverage) + MMR)
        short_liq_price = price * (1 + (1.0 / leverage) - MMR)
        long_idx = np.digitize(long_liq_price, bins_edges) - 1
        short_idx = np.digitize(short_liq_price, bins_edges) - 1
        volume_scale = volume_weights.to_numpy() * open_interest * weight * 1e6
        if 0 <= long_idx < len(bin_centers):
            density[long_idx] += volume_scale[long_idx] * long_ratio
        if 0 <= short_idx < len(bin_centers):
            density[short_idx] += volume_scale[short_idx] * short_ratio

    density = gaussian_filter1d(density, sigma=feature_config.liquidity_smoothing_sigma)
    density_df = pd.DataFrame({"price": bin_centers, "liq_density": density})
    return LiquiditySnapshot(price=price, density=density_df)


def compute_liquidity_features(
    fetcher: BinanceDataFetcher,
    *,
    feature_config: FeatureConfig = FEATURES,
) -> Dict[str, float]:
    snapshot = _coinglass_style_liquidity(fetcher, feature_config=feature_config)
    return snapshot.summary(top_k=feature_config.liquidity_top_k)


__all__ = ["LiquiditySnapshot", "compute_liquidity_features"]
