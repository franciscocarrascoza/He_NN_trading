from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Tuple

import yaml


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


@dataclass(frozen=True)
class BinanceAPIConfig:
    symbol: str
    interval: str
    history_limit: int
    max_klines_per_request: int
    futures_base_url: str
    spot_base_url: str
    order_book_limit: int
    long_short_period: str
    liquidation_bins: int
    liquidation_price_range: float


@dataclass(frozen=True)
class DataConfig:
    feature_window: int
    forecast_horizon: int
    validation_split: float
    use_extras: bool
    extras_time_aligned: bool
    momentum_windows: Tuple[int, ...] = field(default_factory=lambda: (8, 24, 64))
    volatility_windows: Tuple[int, ...] = field(default_factory=lambda: (8, 24, 64))
    autocorr_window: int = 32
    drawdown_window: int = 64


@dataclass(frozen=True)
class FeatureConfig:
    liquidity_bins: int
    liquidity_price_range: float
    liquidity_smoothing_sigma: float
    liquidity_top_k: int
    order_book_depth: int
    add_ema: bool = True
    ema_windows: Tuple[int, ...] = field(default_factory=lambda: (6, 12, 24))


@dataclass(frozen=True)
class ModelConfig:
    hermite_degree: int
    hermite_maps_a: int
    hermite_maps_b: int
    hermite_hidden_dim: int
    dropout: float
    use_lstm: bool = True
    lstm_hidden: int = 64
    hermite_version: str = "probabilist"
    prob_source: Literal["cdf", "logit"] = "cdf"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    gradient_clip: float
    lambda_bce: float
    seed: int
    device_preference: str
    classification_loss: str = "bce"
    auto_pos_weight: bool = True
    focal_gamma: float = 2.0
    min_pos_weight_samples: int = 128
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    scheduler_warmup_pct: float = 0.1
    enable_lr_range_test: bool = False
    lr_range_min: float = 1e-5
    lr_range_max: float = 1e-1
    lr_range_steps: int = 60
    enable_class_downsample: bool = False
    reg_weight: float = 1.0
    cls_weight: float = 1.0
    unc_weight: float = 0.5
    sign_hinge_weight: float = 0.05
    use_cv: bool = True
    cv_folds: int = 5
    early_stop_metric: Literal["AUC", "DirAcc"] = "AUC"
    patience: int = 10
    min_delta: float = 1e-4


@dataclass(frozen=True)
class StrategyConfig:
    thresholds: Tuple[float, ...] = field(default_factory=lambda: (0.55, 0.6))
    confidence_margin: float = 0.1
    kelly_clip: float = 0.5
    use_conformal_gate: bool = True
    conformal_p_min: float = 0.05


@dataclass(frozen=True)
class EvaluationConfig:
    alpha: float
    cv_folds: int
    val_block: int
    calibration_fraction: float
    threshold: float
    cost_bps: float
    save_markdown: bool
    n_bins: int = 15


@dataclass(frozen=True)
class ReportingConfig:
    output_dir: str
    legend_title: str
    date_format: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    binance: BinanceAPIConfig
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    strategy: StrategyConfig
    evaluation: EvaluationConfig
    reporting: ReportingConfig
    logging: LoggingConfig

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AppConfig":
        try:
            binance = BinanceAPIConfig(**mapping["binance"])
            data = DataConfig(**mapping["data"])
            features = FeatureConfig(**mapping["features"])
            model = ModelConfig(**mapping["model"])
            training = TrainingConfig(**mapping["training"])
            evaluation = EvaluationConfig(**mapping["evaluation"])
            reporting = ReportingConfig(**mapping["reporting"])
            logging = LoggingConfig(**mapping.get("logging", {}))
            strategy = StrategyConfig(**mapping["strategy"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Missing required config section: {exc}") from exc
        return cls(
            binance=binance,
            data=data,
            features=features,
            model=model,
            training=training,
            strategy=strategy,
            evaluation=evaluation,
            reporting=reporting,
            logging=logging,
        )


def load_config(
    path: Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    base_path = Path(__file__).with_name("defaults.yaml")
    merged: Dict[str, Any] = _load_yaml(base_path)
    if path is not None:
        external = _load_yaml(path)
        _deep_update(merged, external)
    if overrides:
        _deep_update(merged, overrides)
    return AppConfig.from_mapping(merged)


APP_CONFIG = load_config()
BINANCE = APP_CONFIG.binance
DATA = APP_CONFIG.data
FEATURES = APP_CONFIG.features
MODEL = APP_CONFIG.model
TRAINING = APP_CONFIG.training
EVALUATION = APP_CONFIG.evaluation
REPORTING = APP_CONFIG.reporting
LOGGING = APP_CONFIG.logging
STRATEGY = APP_CONFIG.strategy

__all__ = [
    "AppConfig",
    "BINANCE",
    "DATA",
    "FEATURES",
    "MODEL",
    "TRAINING",
    "EVALUATION",
    "REPORTING",
    "LOGGING",
    "BinanceAPIConfig",
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "TrainingConfig",
    "StrategyConfig",
    "EvaluationConfig",
    "ReportingConfig",
    "LoggingConfig",
    "load_config",
]
