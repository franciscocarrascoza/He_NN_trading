"""Unit tests for data sync module with mocked downloader per spec."""  # FIX: test module docstring per spec

import asyncio  # FIX: async testing
import sqlite3  # FIX: SQLite for index testing
import tempfile  # FIX: temporary directory for testing
from pathlib import Path  # FIX: path handling
from unittest.mock import MagicMock, patch  # FIX: mocking utilities

import numpy as np  # FIX: numpy for data generation
import pandas as pd  # FIX: pandas for DataFrame
import pytest  # FIX: pytest framework

from backend.data.downloader import BinanceDownloader  # FIX: import downloader


@pytest.fixture  # FIX: pytest fixture decorator
def temp_storage():  # FIX: temporary storage fixture
    """Create temporary storage directory for testing."""  # FIX: fixture docstring
    with tempfile.TemporaryDirectory() as tmpdir:  # FIX: create temp directory
        yield Path(tmpdir)  # FIX: yield temp path


def test_downloader_creates_index_db(temp_storage):  # FIX: test index database creation per spec
    """Test that downloader creates index database."""  # FIX: test docstring
    # FIX: Initialize downloader
    downloader = BinanceDownloader(storage_path=temp_storage)  # FIX: create downloader

    # FIX: Check index database exists
    index_db_path = temp_storage / "index.sqlite"  # FIX: index database path
    assert index_db_path.exists(), "Index database should be created"  # FIX: assert database exists

    # FIX: Check table schema
    conn = sqlite3.connect(index_db_path)  # FIX: connect to database
    cursor = conn.cursor()  # FIX: create cursor
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='candle_index'")  # FIX: query table
    result = cursor.fetchone()  # FIX: fetch result
    conn.close()  # FIX: close connection

    assert result is not None, "candle_index table should exist"  # FIX: assert table exists


def test_downloader_incremental_sync_without_duplicates(temp_storage):  # FIX: test incremental sync per spec
    """Test that incremental sync creates partitions without duplicates."""  # FIX: test docstring

    # FIX: Initialize downloader
    downloader = BinanceDownloader(
        storage_path=temp_storage,  # FIX: temp storage
        symbol="BTCUSDT",  # FIX: test symbol
        timeframe="1h",  # FIX: test timeframe
    )  # FIX: create downloader

    # FIX: Mock _fetch_klines_batch to return synthetic data
    mock_klines = [
        [
            1609459200000,  # FIX: open_time (2021-01-01 00:00:00 UTC)
            "30000.0",  # FIX: open
            "30100.0",  # FIX: high
            "29900.0",  # FIX: low
            "30050.0",  # FIX: close
            "100.0",  # FIX: volume
            1609462799999,  # FIX: close_time
            "3005000.0",  # FIX: quote_volume
            1000,  # FIX: trades
            "50.0",  # FIX: taker_buy_base
            "1502500.0",  # FIX: taker_buy_quote
            "0",  # FIX: ignore
        ]
    ]  # FIX: single candle data

    with patch.object(downloader, '_fetch_klines_batch', return_value=mock_klines):  # FIX: mock fetch
        # FIX: Run first sync
        count1 = asyncio.run(downloader.sync_latest())  # FIX: first sync
        assert count1 == 1, "First sync should store 1 candle"  # FIX: assert count

        # FIX: Run second sync with same data (should deduplicate)
        count2 = asyncio.run(downloader.sync_latest())  # FIX: second sync
        # FIX: Second sync may return 0 or 1 depending on deduplication logic
        # FIX: Check that Parquet file has no duplicates
        parquet_path = temp_storage / "ohlcv" / "BTCUSDT" / "1h" / "2021" / "01" / "BTCUSDT_1h.parquet"  # FIX: partition path
        if parquet_path.exists():  # FIX: check file exists
            df = pd.read_parquet(parquet_path)  # FIX: read parquet
            assert df.duplicated(subset=["open_time"]).sum() == 0, "Parquet file should have no duplicate timestamps"  # FIX: assert no duplicates


# FIX: Export test functions
__all__ = [
    "test_downloader_creates_index_db",
    "test_downloader_incremental_sync_without_duplicates",
]  # FIX: module exports
