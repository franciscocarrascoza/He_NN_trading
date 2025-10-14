from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BinanceAPIConfig:
    """Global configuration for Binance HTTP endpoints and data windows."""

    symbol: str = "BTCUSDT"
    interval: str = "1h"
    history_limit: int = 1500
    max_klines_per_request: int = 1500
    futures_base_url: str = "https://fapi.binance.com"
    spot_base_url: str = "https://api.binance.com"
    order_book_limit: int = 100
    long_short_period: str = "5m"
    liquidation_bins: int = 200
    liquidation_price_range: float = 0.75


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering."""

    liquidity_bins: int = 180
    liquidity_price_range: float = 0.6
    liquidity_smoothing_sigma: float = 5.0
    liquidity_top_k: int = 10
    order_book_depth: int = 20


@dataclass(frozen=True)
class TrainingConfig:
    """High level hyper-parameters for the Hermite forecaster."""

    forecast_horizon: int = 1  # number of candles ahead to predict
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    hermite_degree: int = 5
    hermite_maps_a: int = 2
    hermite_maps_b: int = 2
    hermite_hidden_dim: int = 32
    jacobian_mode: str = "summary"
    feature_window: int = 64
    validation_split: float = 0.2
    random_seed: int = 42
    device_preference: str = "RTX 2060"
    checkpoint_path: str = "artifacts/hermite_forecaster.pt"

    @property
    def seed(self) -> int:
        """Alias for backwards compatibility with legacy ``random_seed`` usages."""

        return self.random_seed


BINANCE = BinanceAPIConfig()
FEATURES = FeatureConfig()
TRAINING = TrainingConfig()

__all__: Tuple[str, ...] = ("BINANCE", "FEATURES", "TRAINING", "BinanceAPIConfig", "FeatureConfig", "TrainingConfig")
