"""Incremental Binance historical data downloader with rate limiting and Parquet storage."""  # FIX: downloader module per spec

from __future__ import annotations  # FIX: modern type hint compatibility

import asyncio  # FIX: async HTTP and rate limiting
import logging  # FIX: structured logging
import math  # FIX: pagination calculations
import sqlite3  # FIX: metadata index storage
import time  # FIX: rate limit token bucket timing
from datetime import datetime, timezone  # FIX: timestamp handling
from pathlib import Path  # FIX: cross-platform path handling
from typing import Any, Dict, List, Optional, Tuple  # FIX: type annotations

import numpy as np  # FIX: array handling
import pandas as pd  # FIX: DataFrame and Parquet I/O
import requests  # FIX: synchronous HTTP fallback

# FIX: Configure logging per spec
LOGGER = logging.getLogger(__name__)  # FIX: module-level logger


class TokenBucket:
    """Token bucket rate limiter for API calls."""  # FIX: rate limiting helper per spec

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialize token bucket.

        Args:
            rate: Tokens per second refill rate  # FIX: refill rate param
            capacity: Maximum bucket capacity  # FIX: max tokens param
        """  # FIX: constructor docstring
        self.rate = rate  # FIX: store refill rate
        self.capacity = capacity  # FIX: store capacity
        self.tokens = capacity  # FIX: initialize to full capacity
        self.last_refill = time.monotonic()  # FIX: track last refill time

    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens, return True if successful."""  # FIX: token consumption logic
        self._refill()  # FIX: refill tokens based on elapsed time
        if self.tokens >= tokens:  # FIX: check if enough tokens available
            self.tokens -= tokens  # FIX: consume tokens
            return True  # FIX: consumption successful
        return False  # FIX: insufficient tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""  # FIX: refill helper
        now = time.monotonic()  # FIX: current time
        elapsed = now - self.last_refill  # FIX: time since last refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)  # FIX: add tokens up to capacity
        self.last_refill = now  # FIX: update last refill time


class BinanceDownloader:
    """Incremental Binance historical candle downloader with Parquet storage."""  # FIX: downloader class per spec

    # FIX: Class constants for Binance API and rate limiting
    BASE_URL = "https://fapi.binance.com"  # FIX: Binance Futures API base URL
    KLINES_ENDPOINT = "/fapi/v1/klines"  # FIX: klines endpoint path
    MAX_KLINES_PER_REQUEST = 1500  # FIX: Binance API limit
    RATE_LIMIT_TOKENS_PER_SEC = 10.0  # FIX: conservative rate limit tokens/sec
    RATE_LIMIT_CAPACITY = 50.0  # FIX: token bucket capacity

    # FIX: Timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60 * 1000,  # FIX: 1 minute
        "3m": 3 * 60 * 1000,  # FIX: 3 minutes
        "5m": 5 * 60 * 1000,  # FIX: 5 minutes
        "15m": 15 * 60 * 1000,  # FIX: 15 minutes
        "30m": 30 * 60 * 1000,  # FIX: 30 minutes
        "1h": 60 * 60 * 1000,  # FIX: 1 hour
        "2h": 2 * 60 * 60 * 1000,  # FIX: 2 hours
        "4h": 4 * 60 * 60 * 1000,  # FIX: 4 hours
        "6h": 6 * 60 * 60 * 1000,  # FIX: 6 hours
        "8h": 8 * 60 * 60 * 1000,  # FIX: 8 hours
        "12h": 12 * 60 * 60 * 1000,  # FIX: 12 hours
        "1d": 24 * 60 * 60 * 1000,  # FIX: 1 day
        "3d": 3 * 24 * 60 * 60 * 1000,  # FIX: 3 days
        "1w": 7 * 24 * 60 * 60 * 1000,  # FIX: 1 week
    }  # FIX: timeframe mapping

    def __init__(
        self,
        storage_path: Path,  # FIX: base storage directory
        symbol: str = "BTCUSDT",  # FIX: trading pair symbol
        timeframe: str = "1h",  # FIX: candle timeframe
        max_backfill_years: int = 8,  # FIX: maximum backfill window
    ) -> None:
        """Initialize Binance downloader.

        Args:
            storage_path: Base directory for Parquet storage  # FIX: storage param
            symbol: Trading pair symbol  # FIX: symbol param
            timeframe: Candle timeframe (1m, 5m, 1h, etc.)  # FIX: timeframe param
            max_backfill_years: Maximum years to backfill  # FIX: backfill limit param
        """  # FIX: constructor docstring
        self.storage_path = Path(storage_path)  # FIX: store base path
        self.symbol = symbol  # FIX: store symbol
        self.timeframe = timeframe  # FIX: store timeframe
        self.max_backfill_years = max_backfill_years  # FIX: store backfill limit
        self.token_bucket = TokenBucket(
            rate=self.RATE_LIMIT_TOKENS_PER_SEC,  # FIX: initialize rate limiter
            capacity=self.RATE_LIMIT_CAPACITY,  # FIX: set capacity
        )  # FIX: create token bucket

        # FIX: Ensure storage and index directories exist
        self.storage_path.mkdir(parents=True, exist_ok=True)  # FIX: create storage directory
        self.index_db_path = self.storage_path / "index.sqlite"  # FIX: index database path
        self._init_index_db()  # FIX: initialize index database

    def _init_index_db(self) -> None:
        """Initialize SQLite index database for metadata."""  # FIX: index init per spec
        conn = sqlite3.connect(self.index_db_path)  # FIX: connect to SQLite
        cursor = conn.cursor()  # FIX: create cursor
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS candle_index (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                latest_timestamp_ms INTEGER NOT NULL,
                last_updated_utc TEXT NOT NULL,
                PRIMARY KEY (symbol, timeframe)
            )
            """  # FIX: create index table schema per spec
        )  # FIX: execute table creation
        conn.commit()  # FIX: commit transaction
        conn.close()  # FIX: close connection
        LOGGER.info("Initialized index database", extra={"event": "index_db_init", "path": str(self.index_db_path)})  # FIX: log init

    def _get_latest_timestamp(self) -> Optional[int]:
        """Retrieve latest stored timestamp for symbol/timeframe."""  # FIX: query latest timestamp per spec
        conn = sqlite3.connect(self.index_db_path)  # FIX: connect to SQLite
        cursor = conn.cursor()  # FIX: create cursor
        cursor.execute(
            "SELECT latest_timestamp_ms FROM candle_index WHERE symbol = ? AND timeframe = ?",  # FIX: query index
            (self.symbol, self.timeframe),  # FIX: query parameters
        )  # FIX: execute query
        row = cursor.fetchone()  # FIX: fetch result
        conn.close()  # FIX: close connection
        return int(row[0]) if row else None  # FIX: return timestamp or None

    def _update_latest_timestamp(self, timestamp_ms: int) -> None:
        """Update latest timestamp in index."""  # FIX: update index per spec
        conn = sqlite3.connect(self.index_db_path)  # FIX: connect to SQLite
        cursor = conn.cursor()  # FIX: create cursor
        now_utc = datetime.now(timezone.utc).isoformat()  # FIX: current UTC timestamp
        cursor.execute(
            """
            INSERT INTO candle_index (symbol, timeframe, latest_timestamp_ms, last_updated_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe) DO UPDATE SET
                latest_timestamp_ms = excluded.latest_timestamp_ms,
                last_updated_utc = excluded.last_updated_utc
            """,  # FIX: upsert latest timestamp
            (self.symbol, self.timeframe, timestamp_ms, now_utc),  # FIX: upsert parameters
        )  # FIX: execute upsert
        conn.commit()  # FIX: commit transaction
        conn.close()  # FIX: close connection

    def _fetch_klines_batch(
        self, start_ms: int, end_ms: int, max_retries: int = 4  # FIX: exponential backoff retries per spec
    ) -> List[List[Any]]:
        """Fetch a single batch of klines with rate limiting and retry logic."""  # FIX: batch fetch with retries per spec
        url = f"{self.BASE_URL}{self.KLINES_ENDPOINT}"  # FIX: build endpoint URL
        params = {
            "symbol": self.symbol,  # FIX: symbol parameter
            "interval": self.timeframe,  # FIX: interval parameter
            "startTime": start_ms,  # FIX: start time parameter
            "endTime": end_ms,  # FIX: end time parameter
            "limit": self.MAX_KLINES_PER_REQUEST,  # FIX: limit parameter
        }  # FIX: request parameters

        for attempt in range(max_retries):  # FIX: retry loop per spec
            # FIX: Wait for rate limit token
            while not self.token_bucket.consume(1.0):  # FIX: token bucket consumption
                time.sleep(0.1)  # FIX: sleep if no tokens available

            try:
                response = requests.get(url, params=params, timeout=10)  # FIX: HTTP GET request
                response.raise_for_status()  # FIX: raise on HTTP error
                data = response.json()  # FIX: parse JSON response
                LOGGER.debug(
                    "Fetched klines batch",
                    extra={
                        "event": "klines_batch_fetched",
                        "count": len(data),
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    },
                )  # FIX: log batch fetch
                return data  # FIX: return klines data
            except requests.exceptions.RequestException as exc:  # FIX: handle HTTP errors
                backoff_seconds = 2 ** attempt  # FIX: exponential backoff per spec
                LOGGER.warning(
                    "Klines fetch failed, retrying",
                    extra={
                        "event": "klines_fetch_retry",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "backoff_seconds": backoff_seconds,
                        "error": str(exc),
                    },
                )  # FIX: log retry
                if attempt < max_retries - 1:  # FIX: check if more retries available
                    time.sleep(backoff_seconds)  # FIX: sleep before retry
                else:
                    raise  # FIX: re-raise on final attempt

        return []  # FIX: return empty on exhausted retries

    def _parse_and_store_klines(self, klines: List[List[Any]]) -> int:
        """Parse klines and store to Parquet with row group partitioning."""  # FIX: Parquet storage per spec
        if not klines:  # FIX: handle empty input
            return 0  # FIX: no candles to store

        # FIX: Parse klines to DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],  # FIX: column names per Binance API spec
        )  # FIX: create DataFrame

        # FIX: Convert to numeric types
        df["open_time"] = pd.to_numeric(df["open_time"], downcast="integer")  # FIX: timestamp to int
        df["close_time"] = pd.to_numeric(df["close_time"], downcast="integer")  # FIX: timestamp to int
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:  # FIX: price/volume columns
            df[col] = pd.to_numeric(df[col], downcast="float")  # FIX: convert to float
        df["trades"] = pd.to_numeric(df["trades"], downcast="integer")  # FIX: trades count to int

        # FIX: Add datetime columns for partitioning
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)  # FIX: datetime from timestamp
        df["year"] = df["timestamp"].dt.year  # FIX: year partition key
        df["month"] = df["timestamp"].dt.month  # FIX: month partition key

        # FIX: Data QC per spec: detect duplicates and NaNs
        duplicates = df.duplicated(subset=["open_time"], keep="first")  # FIX: find duplicate timestamps
        if duplicates.any():  # FIX: handle duplicates
            dup_count = duplicates.sum()  # FIX: count duplicates
            LOGGER.warning(
                "Dropping duplicate timestamps",
                extra={"event": "data_qc_duplicates", "count": dup_count},  # FIX: log duplicate warning
            )
            df = df[~duplicates]  # FIX: drop duplicates

        # FIX: Check for NaN values
        nan_counts = df.isna().sum()  # FIX: count NaNs per column
        if nan_counts.any():  # FIX: handle NaNs
            LOGGER.warning(
                "NaN values detected in candle data",
                extra={"event": "data_qc_nans", "nan_counts": nan_counts.to_dict()},  # FIX: log NaN warning
            )
            # FIX: For small gaps, forward fill; for large gaps, flag (simplified: drop NaNs)
            df = df.dropna()  # FIX: drop NaNs for now

        # FIX: Partition by year and month per spec
        for (year, month), group in df.groupby(["year", "month"]):  # FIX: iterate partitions
            partition_dir = self.storage_path / "ohlcv" / self.symbol / self.timeframe / str(year) / f"{month:02d}"  # FIX: partition directory path per spec
            partition_dir.mkdir(parents=True, exist_ok=True)  # FIX: create partition directory
            partition_file = partition_dir / f"{self.symbol}_{self.timeframe}.parquet"  # FIX: partition file name

            # FIX: Append or create Parquet file
            if partition_file.exists():  # FIX: check if partition file exists
                existing_df = pd.read_parquet(partition_file)  # FIX: read existing data
                combined_df = pd.concat([existing_df, group], ignore_index=True)  # FIX: combine with new data
                combined_df = combined_df.drop_duplicates(subset=["open_time"], keep="last")  # FIX: deduplicate
                combined_df = combined_df.sort_values("open_time")  # FIX: sort by timestamp
                combined_df.to_parquet(partition_file, index=False, engine="pyarrow")  # FIX: write Parquet
            else:
                group.to_parquet(partition_file, index=False, engine="pyarrow")  # FIX: write new Parquet file

        return len(df)  # FIX: return count of stored candles

    async def download_range(self, start_ts: int, end_ts: int) -> int:
        """Download klines for a specific time range."""  # FIX: range download per spec
        LOGGER.info(
            "Starting range download",
            extra={
                "event": "download_range_start",
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
        )  # FIX: log range download start

        tf_ms = self.TIMEFRAME_MS.get(self.timeframe, 3600 * 1000)  # FIX: get timeframe milliseconds
        total_stored = 0  # FIX: total candles stored counter

        current_start = start_ts  # FIX: current batch start
        while current_start < end_ts:  # FIX: pagination loop
            batch_end = min(current_start + tf_ms * self.MAX_KLINES_PER_REQUEST, end_ts)  # FIX: calculate batch end
            klines = self._fetch_klines_batch(current_start, batch_end)  # FIX: fetch batch
            if not klines:  # FIX: handle empty batch
                break  # FIX: exit loop on empty batch

            stored_count = self._parse_and_store_klines(klines)  # FIX: parse and store
            total_stored += stored_count  # FIX: increment counter

            # FIX: Update index with latest timestamp
            if klines:  # FIX: update index if data fetched
                latest_ts = int(klines[-1][0])  # FIX: get latest timestamp from batch
                self._update_latest_timestamp(latest_ts)  # FIX: update index

            current_start = batch_end  # FIX: advance to next batch

        LOGGER.info(
            "Range download complete",
            extra={
                "event": "download_range_complete",
                "total_stored": total_stored,
            },
        )  # FIX: log range download complete

        return total_stored  # FIX: return total stored count

    async def sync_latest(self) -> int:
        """Incremental sync of latest candles."""  # FIX: incremental sync per spec
        LOGGER.info(
            "Starting incremental sync",
            extra={
                "event": "sync_latest_start",
                "symbol": self.symbol,
                "timeframe": self.timeframe,
            },
        )  # FIX: log sync start

        latest_ts = self._get_latest_timestamp()  # FIX: get latest stored timestamp
        if latest_ts is None:  # FIX: no data stored, perform backfill
            # FIX: Calculate backfill start time
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  # FIX: current time in ms
            backfill_ms = self.max_backfill_years * 365 * 24 * 60 * 60 * 1000  # FIX: backfill window
            start_ts = now_ms - backfill_ms  # FIX: backfill start timestamp
            return await self.download_range(start_ts, now_ms)  # FIX: download range

        # FIX: Incremental update from latest timestamp
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  # FIX: current time in ms
        return await self.download_range(latest_ts + 1, now_ms)  # FIX: download from latest + 1 to now


# FIX: Export downloader class
__all__ = ["BinanceDownloader"]  # FIX: module exports
