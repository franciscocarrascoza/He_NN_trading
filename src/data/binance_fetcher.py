from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests import exceptions as req_exc

from src.config import BINANCE, BinanceAPIConfig

logger = logging.getLogger(__name__)

CANDLE_COLUMNS: Tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)


class BinanceDataFetcher:
    """Lightweight HTTP client tailored for the Binance Futures REST API."""

    def __init__(self, config: BinanceAPIConfig = BINANCE, session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    def _get(self, path: str, *, params: Optional[Dict[str, str]] = None, futures: bool = True) -> Dict:
        base = self.config.futures_base_url if futures else self.config.spot_base_url
        response = self.session.get(f"{base}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _klines_to_frame(raw_klines: List[List]) -> pd.DataFrame:
        if not isinstance(raw_klines, list) or not raw_klines:
            raise ValueError("Expected non-empty list for kline payload.")

        df = pd.DataFrame(raw_klines, columns=CANDLE_COLUMNS)
        numeric_columns = [c for c in CANDLE_COLUMNS if c not in ("open_time", "close_time", "ignore")]
        df[numeric_columns] = df[numeric_columns].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df.sort_values("close_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _synthetic_klines(self, limit: int) -> pd.DataFrame:
        if limit <= 0:
            raise ValueError("limit must be positive")

        try:
            freq = pd.to_timedelta(self.config.interval)
        except ValueError:
            freq = pd.Timedelta(minutes=1)
        if freq <= pd.Timedelta(0):
            freq = pd.Timedelta(minutes=1)

        end = pd.Timestamp.now(tz="UTC")
        close_times = pd.date_range(end=end, periods=limit, freq=freq)
        open_times = close_times - freq

        indices = np.arange(limit, dtype=float)
        base_price = 25_000.0
        trend = np.linspace(-50.0, 50.0, limit)
        seasonal = 75.0 * np.sin(indices / max(limit, 1) * 6.0)
        close = base_price + trend + seasonal
        open_ = close + 10.0 * np.sin(indices / max(limit, 1) * 3.0)
        high = np.maximum(open_, close) + 12.5
        low = np.minimum(open_, close) - 12.5
        volume = np.linspace(500.0, 650.0, limit)
        quote_asset_volume = volume * ((open_ + close) / 2.0)
        taker_buy_base = volume * 0.55
        taker_buy_quote = quote_asset_volume * 0.55
        number_of_trades = np.linspace(200, 400, limit, dtype=np.int64)

        open_ms = (open_times.view(np.int64) // 1_000_000).astype(np.int64)
        close_ms = (close_times.view(np.int64) // 1_000_000).astype(np.int64)

        rows: List[List[float]] = []
        for idx in range(limit):
            rows.append(
                [
                    int(open_ms[idx]),
                    float(open_[idx]),
                    float(high[idx]),
                    float(low[idx]),
                    float(close[idx]),
                    float(volume[idx]),
                    int(close_ms[idx]),
                    float(quote_asset_volume[idx]),
                    int(number_of_trades[idx]),
                    float(taker_buy_base[idx]),
                    float(taker_buy_quote[idx]),
                    0.0,
                ]
            )

        return self._klines_to_frame(rows)

    def _synthetic_order_book(self, depth: int) -> Dict[str, List[Tuple[float, float]]]:
        candles = self.get_historical_candles(limit=max(depth + 1, 10))
        mid_price = float(candles["close"].iloc[-1])
        price_step = max(mid_price * 0.0005, 0.5)
        base_qty = 5.0

        bids: List[Tuple[float, float]] = []
        asks: List[Tuple[float, float]] = []
        for level in range(depth):
            offset = (level + 1) * price_step
            qty = base_qty * float(np.exp(-level / max(depth, 1)))
            bids.append((mid_price - offset, qty))
            asks.append((mid_price + offset, qty))

        return {"bids": bids, "asks": asks, "last_update_id": 0}

    @staticmethod
    def _synthetic_long_short_ratio(limit: int) -> List[Dict[str, str]]:
        now_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
        ratio = 0.55
        entries = []
        for _ in range(limit):
            entries.append(
                {
                    "timestamp": str(now_ms),
                    "longAccount": f"{ratio:.4f}",
                    "shortAccount": f"{1.0 - ratio:.4f}",
                }
            )
        return entries

    @staticmethod
    def _synthetic_funding_rate(limit: int) -> List[Dict[str, str]]:
        now_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
        entries = []
        for _ in range(limit):
            entries.append({"fundingRate": f"{0.0001:.6f}", "fundingTime": str(now_ms)})
        return entries

    # ------------------------------------------------------------------
    # Public API
    def get_historical_candles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Return the most recent historical candles as a DataFrame."""

        limit = limit or self.config.history_limit
        params = {
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "limit": limit,
        }
        try:
            raw_klines: List[List] = self._get("/fapi/v1/klines", params=params)
            return self._klines_to_frame(raw_klines)
        except (req_exc.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Falling back to synthetic candles: %s", exc)
            return self._synthetic_klines(limit)

    def get_order_book(self, depth: Optional[int] = None) -> Dict[str, List[Tuple[float, float]]]:
        """Retrieve the current futures order book up to the requested depth."""

        depth = depth or self.config.order_book_limit
        params = {"symbol": self.config.symbol, "limit": depth}
        try:
            snapshot = self._get("/fapi/v1/depth", params=params)
            bids = [(float(price), float(qty)) for price, qty in snapshot.get("bids", [])]
            asks = [(float(price), float(qty)) for price, qty in snapshot.get("asks", [])]
            return {"bids": bids, "asks": asks, "last_update_id": snapshot.get("lastUpdateId")}
        except (req_exc.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Falling back to synthetic order book: %s", exc)
            return self._synthetic_order_book(depth)

    def get_mark_price(self) -> float:
        try:
            response = self._get("/fapi/v1/premiumIndex", params={"symbol": self.config.symbol})
            return float(response.get("markPrice", 0.0))
        except (req_exc.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Falling back to synthetic mark price: %s", exc)
            candles = self.get_historical_candles(limit=1)
            return float(candles["close"].iloc[-1])

    def get_open_interest(self) -> float:
        try:
            response = self._get("/fapi/v1/openInterest", params={"symbol": self.config.symbol})
            return float(response.get("openInterest", 0.0))
        except (req_exc.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Falling back to synthetic open interest: %s", exc)
            candles = self.get_historical_candles(limit=24)
            return float(candles["volume"].tail(24).mean())

    def get_long_short_ratio(self, limit: int = 1) -> List[Dict[str, str]]:
        params = {
            "symbol": self.config.symbol,
            "period": self.config.long_short_period,
            "limit": limit,
        }
        try:
            return self._get("/futures/data/globalLongShortAccountRatio", params=params)
        except (req_exc.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Falling back to synthetic long/short ratio: %s", exc)
            return self._synthetic_long_short_ratio(limit)

    def get_funding_rate(self, limit: int = 1) -> List[Dict[str, str]]:
        params = {"symbol": self.config.symbol, "limit": limit}
        try:
            return self._get("/fapi/v1/fundingRate", params=params)
        except (req_exc.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Falling back to synthetic funding rate: %s", exc)
            return self._synthetic_funding_rate(limit)

    def get_volatility_estimate(self, lookback: int = 24) -> float:
        candles = self.get_historical_candles(limit=lookback)
        close_pct = candles["close"].pct_change().dropna()
        return float(close_pct.std())

    def now(self) -> dt.datetime:
        """Return current UTC time (wrapper kept for convenience/testing)."""

        return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


__all__ = ["BinanceDataFetcher"]
