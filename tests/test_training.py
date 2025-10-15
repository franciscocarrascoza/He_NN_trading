from __future__ import annotations

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from src.config import TRAINING
from src.data.dataset import HermiteDataset
from src.pipeline.training import HermiteTrainer
from src.features import compute_liquidity_features_series, compute_orderbook_features_series


def _make_candles(count: int) -> pd.DataFrame:
    times = np.arange(count, dtype=np.int64) * 60_000
    data = {
        "open": np.linspace(100.0, 100.0 + count - 1, count),
        "high": np.linspace(101.0, 101.0 + count - 1, count),
        "low": np.linspace(99.0, 99.0 + count - 1, count),
        "close": np.linspace(100.0, 100.0 + count - 1, count),
        "volume": np.linspace(10.0, 10.0 + count - 1, count),
        "quote_asset_volume": np.linspace(1000.0, 1200.0, count),
        "number_of_trades": np.linspace(50, 80, count),
        "taker_buy_base": np.linspace(5.0, 7.0, count),
        "taker_buy_quote": np.linspace(500.0, 600.0, count),
        "close_time": times,
    }
    return pd.DataFrame(data)


def _make_dataset(num_candles: int = 40, window: int = 5, horizon: int = 2) -> HermiteDataset:
    candles = _make_candles(num_candles)
    liquidity_series = compute_liquidity_features_series(candles)
    orderbook_series = compute_orderbook_features_series(candles)
    return HermiteDataset(
        candles,
        liquidity_features=liquidity_series,
        orderbook_features=orderbook_series,
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
    assert "direction_prob" in frame.columns
    assert np.all((frame["direction_prob"].to_numpy() >= 0.0) & (frame["direction_prob"].to_numpy() <= 1.0))
    assert np.all(frame["direction_hit"].isin([0, 1]))


def test_dataset_uses_causal_windows() -> None:
    window = 4
    horizon = 3
    num_candles = 32
    dataset = _make_dataset(num_candles=num_candles, window=window, horizon=horizon)
    base_names = HermiteDataset.base_feature_names
    close_idx = base_names.index("close")
    close_times = dataset.candles["close_time"].to_numpy()
    values = dataset.candles["close"].to_numpy()
    for idx in range(len(dataset)):
        anchor_idx = window - 1 + idx
        target_idx = anchor_idx + horizon
        item = dataset.item(idx)
        assert item.anchor_time.item() == close_times[anchor_idx]
        assert item.target_time.item() == close_times[target_idx]
        window_slice = values[anchor_idx - window + 1:anchor_idx + 1]
        flattened = (
            item.features[: window * len(base_names)]
            .view(window, len(base_names))
            .detach()
            .cpu()
            .numpy()
        )
        # Most recent close in window must match anchor close price
        assert np.isclose(flattened[-1, close_idx], values[anchor_idx])
        # Future close should never appear in the window
        assert not np.isclose(window_slice, values[target_idx]).any()
