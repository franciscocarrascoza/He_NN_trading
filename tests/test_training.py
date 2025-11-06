from __future__ import annotations

import os
from dataclasses import replace

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from src.config import APP_CONFIG, DATA, EVALUATION, REPORTING, TRAINING, DataConfig
from src.data.dataset import HermiteDataset
from src.pipeline import HermiteTrainer
from src.pipeline.scaler import LeakageGuardScaler
from src.pipeline.split import RollingOriginSplitter


def _make_candles(count: int) -> pd.DataFrame:
    times = np.arange(count, dtype=np.int64) * 60_000
    data = {
        "open": np.linspace(100.0, 100.0 + count - 1, count),
        "high": np.linspace(101.0, 101.0 + count - 1, count),
        "low": np.linspace(99.0, 99.0 + count - 1, count),
        "close": np.linspace(100.0, 100.5 + count - 1, count),
        "volume": np.linspace(10.0, 10.0 + count - 1, count),
        "quote_asset_volume": np.linspace(1000.0, 1200.0, count),
        "number_of_trades": np.linspace(50, 80, count),
        "taker_buy_base": np.linspace(5.0, 7.0, count),
        "taker_buy_quote": np.linspace(500.0, 600.0, count),
        "close_time": times,
    }
    return pd.DataFrame(data)


def _make_config(window: int, horizon: int) -> DataConfig:
    return replace(DATA, feature_window=window, forecast_horizon=horizon)


def _make_dataset(num_candles: int = 40, window: int = 5, horizon: int = 2) -> HermiteDataset:
    candles = _make_candles(num_candles)
    config = _make_config(window=window, horizon=horizon)
    return HermiteDataset(candles, data_config=config)


def test_scaler_prevents_future_indices() -> None:
    dataset = _make_dataset()
    scaler_idx = torch.arange(len(dataset) - 5)
    scaler = LeakageGuardScaler(max_index=int(scaler_idx[-1].item()))
    scaler.fit(dataset.features, scaler_idx)

    with pytest.raises(ValueError):
        bad_indices = torch.arange(len(dataset))
        scaler.fit(dataset.features, bad_indices)


def test_scaler_has_nonzero_variance() -> None:
    dataset = _make_dataset()
    scaler_idx = torch.arange(len(dataset) - 5)
    scaler = LeakageGuardScaler(max_index=int(scaler_idx[-1].item()))
    _, stats = scaler.fit_transform(dataset.features, scaler_idx)
    assert torch.all(stats.std > 0.0)


def test_log_return_features_zero_mean_after_scaling() -> None:
    dataset = _make_dataset()
    scaler_idx = torch.arange(len(dataset) - 5)
    scaler = LeakageGuardScaler(max_index=int(scaler_idx[-1].item()))
    scaled_features, _ = scaler.fit_transform(dataset.features, scaler_idx)
    train_features = scaled_features.index_select(0, scaler_idx)
    log_ret_columns = [idx for idx, name in enumerate(dataset.feature_names) if name.startswith("log_ret_close")]
    assert log_ret_columns, "Expected log_ret_close features to be present."
    subset = train_features[:, log_ret_columns]
    means = subset.mean(dim=0)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)


def test_rolling_split_prevents_leakage() -> None:
    dataset = _make_dataset(num_candles=80, window=6, horizon=2)
    data_cfg = _make_config(window=6, horizon=2)
    evaluation_cfg = replace(EVALUATION, val_block=8, cv_folds=1, calibration_fraction=0.1)
    splitter = RollingOriginSplitter(
        dataset_length=len(dataset),
        data_config=data_cfg,
        evaluation_config=evaluation_cfg,
    )
    folds = splitter.split(use_cv=False)
    assert folds, "Expected at least one fold."
    fold = folds[0]
    assert torch.max(fold.train_idx) < torch.min(fold.val_idx)
    assert torch.max(fold.calibration_idx) < torch.min(fold.val_idx)
    assert torch.max(fold.scaler_idx) < torch.min(fold.val_idx)


def test_trainer_guard_on_direction_label_mismatch(tmp_path) -> None:
    candles = _make_candles(60)
    candles["close"] = 100.0 + np.sin(np.arange(60) / 2.0)
    dataset = HermiteDataset(
        candles,
        data_config=_make_config(window=5, horizon=1),
    )
    dataset.direction_labels = 1 - dataset.direction_labels
    data_cfg = _make_config(window=5, horizon=1)
    training_cfg = replace(TRAINING, num_epochs=1, batch_size=8, seed=0, scheduler="none", enable_lr_range_test=False)
    evaluation_cfg = replace(EVALUATION, save_markdown=False, val_block=10, cv_folds=1, calibration_fraction=0.1)
    reporting_cfg = replace(REPORTING, output_dir=str(tmp_path))
    config = replace(
        APP_CONFIG,
        data=data_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        reporting=reporting_cfg,
    )
    trainer = HermiteTrainer(config=config, device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="Direction labels mismatch"):
        trainer.run(dataset=dataset, use_cv=False, results_dir=tmp_path)


def test_training_fails_on_extreme_imbalance_without_mitigation(tmp_path) -> None:
    data_cfg = _make_config(window=4, horizon=1)
    candles = _make_candles(60)
    dataset = HermiteDataset(candles, data_config=data_cfg)
    training_cfg = replace(
        TRAINING,
        num_epochs=1,
        batch_size=8,
        seed=0,
        scheduler="none",
        enable_lr_range_test=False,
        auto_pos_weight=False,
        classification_loss="bce",
        enable_class_downsample=False,
    )
    evaluation_cfg = replace(EVALUATION, save_markdown=False, val_block=10, cv_folds=1, calibration_fraction=0.1)
    reporting_cfg = replace(REPORTING, output_dir=str(tmp_path))
    config = replace(
        APP_CONFIG,
        data=data_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        reporting=reporting_cfg,
    )
    trainer = HermiteTrainer(config=config, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="all-same-sign labels"):
        trainer.run(dataset=dataset, use_cv=False, results_dir=tmp_path)
