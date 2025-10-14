from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.data.dataset import HermiteDataset


def _make_candles(count: int, *, start: int = 0) -> pd.DataFrame:
    times = np.arange(start, start + count, dtype=np.int64) * 60_000
    data = {
        "open": np.linspace(100.0, 100.0 + count - 1, count),
        "high": np.linspace(101.0, 101.0 + count - 1, count),
        "low": np.linspace(99.0, 99.0 + count - 1, count),
        "close": np.linspace(100.0, 100.0 + count - 1, count),
        "volume": np.linspace(10.0, 10.0 + count - 1, count),
        "close_time": times,
    }
    return pd.DataFrame(data)


def test_dataset_enforces_monotonic_close_time() -> None:
    candles = _make_candles(10)
    candles.loc[5:, "close_time"] = candles.loc[5:, "close_time"].iloc[::-1].values
    with pytest.raises(ValueError):
        HermiteDataset(
            candles,
            liquidity_features={"liq": 0.1},
            orderbook_features={"ob": 0.2},
            feature_window=3,
            forecast_horizon=1,
            normalise=False,
        )


def test_targets_and_anchor_alignment() -> None:
    candles = _make_candles(12)
    dataset = HermiteDataset(
        candles,
        liquidity_features={"liq": 0.1},
        orderbook_features={"ob": 0.2},
        feature_window=3,
        forecast_horizon=2,
        normalise=False,
    )
    for idx in range(len(dataset)):
        item = dataset.item(idx)
        anchor_index = np.where(candles["close_time"].to_numpy() == item.anchor_time.item())[0][0]
        future_index = anchor_index + 2
        anchor_close = candles.iloc[anchor_index]["close"]
        future_close = candles.iloc[future_index]["close"]
        expected_target = math.log(future_close / anchor_close)
        assert pytest.approx(item.target.item(), rel=1e-6) == expected_target
        assert item.anchor_price.item() == pytest.approx(anchor_close)
        assert item.target_time.item() > item.anchor_time.item()
