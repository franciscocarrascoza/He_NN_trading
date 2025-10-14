from __future__ import annotations

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from src.config import TRAINING
from src.data.dataset import HermiteDataset
from src.pipeline.training import HermiteTrainer


def _make_candles(count: int) -> pd.DataFrame:
    times = np.arange(count, dtype=np.int64) * 60_000
    data = {
        "open": np.linspace(100.0, 100.0 + count - 1, count),
        "high": np.linspace(101.0, 101.0 + count - 1, count),
        "low": np.linspace(99.0, 99.0 + count - 1, count),
        "close": np.linspace(100.0, 100.0 + count - 1, count),
        "volume": np.linspace(10.0, 10.0 + count - 1, count),
        "close_time": times,
    }
    return pd.DataFrame(data)


def _make_dataset(num_candles: int = 40, window: int = 5, horizon: int = 2) -> HermiteDataset:
    return HermiteDataset(
        _make_candles(num_candles),
        liquidity_features={"liq": 0.5},
        orderbook_features={"ob": 0.25},
        feature_window=window,
        forecast_horizon=horizon,
        normalise=False,
    )


def test_scaler_fit_uses_train_slice_only(tmp_path) -> None:
    dataset = _make_dataset()
    raw_features = dataset.features.clone()
    config = replace(
        TRAINING,
        num_epochs=1,
        batch_size=8,
        validation_split=0.25,
        checkpoint_path=str(tmp_path / "ckpt.pt"),
    )
    trainer = HermiteTrainer(fetcher=None, training_config=config)
    artifacts = trainer.train(dataset=dataset)

    total_len = len(dataset)
    val_len = max(1, int(total_len * config.validation_split))
    train_len = total_len - val_len
    expected_mean = raw_features[:train_len].mean(dim=0)
    expected_std = raw_features[:train_len].std(dim=0, unbiased=False).clamp_min(1e-6)

    assert torch.allclose(artifacts.feature_mean, expected_mean, atol=1e-6)
    assert torch.allclose(artifacts.feature_std, expected_std, atol=1e-6)
    assert artifacts.train_indices[-1] < artifacts.val_indices[0]
    assert artifacts.checkpoint_path.exists()


def test_price_reconstruction_matches_targets(tmp_path) -> None:
    dataset = _make_dataset()
    config = replace(
        TRAINING,
        num_epochs=1,
        batch_size=8,
        validation_split=0.2,
        checkpoint_path=str(tmp_path / "ckpt.pt"),
    )
    trainer = HermiteTrainer(fetcher=None, training_config=config)
    artifacts = trainer.train(dataset=dataset)

    frame = artifacts.forecast_frame
    anchors = frame["anchor_price"].to_numpy()
    true_lr = frame["true_log_return"].to_numpy()
    pred_lr = frame["pred_log_return"].to_numpy()

    true_prices = anchors * np.exp(true_lr)
    pred_prices = anchors * np.exp(pred_lr)
    assert np.allclose(frame["true_price"].to_numpy(), true_prices)
    assert np.allclose(frame["pred_price"].to_numpy(), pred_prices)
    assert np.allclose(frame["abs_error"].to_numpy(), np.abs(pred_prices - true_prices))
