from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from src.config import DATA, DataConfig
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


def _make_config(window: int = 3, horizon: int = 2) -> DataConfig:
    return replace(DATA, feature_window=window, forecast_horizon=horizon)


def test_dataset_enforces_monotonic_close_time() -> None:
    candles = _make_candles(10)
    candles.loc[5:, "close_time"] = candles.loc[5:, "close_time"].iloc[::-1].values
    with pytest.raises(ValueError):
        HermiteDataset(
            candles,
            data_config=_make_config(window=3, horizon=1),
        )


def test_targets_and_anchor_alignment() -> None:
    candles = _make_candles(12)
    dataset = HermiteDataset(
        candles,
        data_config=_make_config(window=3, horizon=2),
    )
    for idx in range(len(dataset)):
        item = dataset.item(idx)
        anchor_index = np.where(candles["close_time"].to_numpy() == item.anchor_time.item())[0][0]
        future_index = anchor_index + 2
        anchor_close = candles.iloc[anchor_index]["close"]
        future_close = candles.iloc[future_index]["close"]
        expected_target = math.log(future_close / anchor_close)
        assert pytest.approx(item.target.item(), rel=1e-6) == expected_target
        expected_label = 1 if expected_target > 0 else 0
        assert int(item.direction_label.item()) == expected_label
        assert item.anchor_price.item() == pytest.approx(anchor_close)
        assert item.target_time.item() > item.anchor_time.item()


def test_direction_labels_match_target_sign() -> None:
    candles = _make_candles(20)
    dataset = HermiteDataset(
        candles,
        data_config=_make_config(window=4, horizon=1),
    )
    targets = dataset.targets.view(-1).numpy()
    labels = dataset.direction_labels.view(-1).numpy()
    assert np.array_equal(labels, (targets > 0.0).astype(np.int64))


def test_label_horizon_alignment() -> None:
    horizon = 2
    candles = _make_candles(30)
    dataset = HermiteDataset(
        candles,
        data_config=_make_config(window=4, horizon=horizon),
    )
    diffs = dataset.target_times - dataset.anchor_times
    assert torch.all(diffs == horizon * 60_000)


def test_class_balance_reasonable() -> None:
    count = 80
    candles = _make_candles(count)
    candles["close"] = 100.0 + np.sin(np.arange(count) / 3.0)
    dataset = HermiteDataset(
        candles,
        data_config=_make_config(window=5, horizon=1),
    )
    labels = dataset.direction_labels.view(-1).numpy()
    frac = labels.mean()
    assert 0.05 < frac < 0.95
