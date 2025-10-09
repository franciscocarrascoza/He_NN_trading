from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.config import BINANCE, BinanceAPIConfig


class BinanceDataFetcher:
    """Lightweight HTTP client tailored for the Binance Futures REST API."""

    def __init__(self, config: BinanceAPIConfig = BINANCE, session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def _get(self, path: str, *, params: Optional[Dict[str, str]] = None, futures: bool = True) -> Dict:
        base = self.config.futures_base_url if futures else self.config.spot_base_url
        response = self.session.get(f"{base}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_historical_candles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Return the most recent historical candles as a DataFrame."""

        limit = limit or self.config.history_limit
        params = {
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "limit": limit,
        }
        raw_klines: List[List] = self._get("/fapi/v1/klines", params=params)
        columns = [
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
        ]
        df = pd.DataFrame(raw_klines, columns=columns)
        numeric_columns = [c for c in columns if c not in ("open_time", "close_time", "ignore")]
        df[numeric_columns] = df[numeric_columns].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    def get_order_book(self, depth: Optional[int] = None) -> Dict[str, List[Tuple[float, float]]]:
        """Retrieve the current futures order book up to the requested depth."""

        depth = depth or self.config.order_book_limit
        params = {"symbol": self.config.symbol, "limit": depth}
        snapshot = self._get("/fapi/v1/depth", params=params)
        bids = [(float(price), float(qty)) for price, qty in snapshot.get("bids", [])]
        asks = [(float(price), float(qty)) for price, qty in snapshot.get("asks", [])]
        return {"bids": bids, "asks": asks, "last_update_id": snapshot.get("lastUpdateId")}

    def get_mark_price(self) -> float:
        response = self._get("/fapi/v1/premiumIndex", params={"symbol": self.config.symbol})
        return float(response.get("markPrice", 0.0))

    def get_open_interest(self) -> float:
        response = self._get("/fapi/v1/openInterest", params={"symbol": self.config.symbol})
        return float(response.get("openInterest", 0.0))

    def get_long_short_ratio(self, limit: int = 1) -> List[Dict[str, str]]:
        params = {
            "symbol": self.config.symbol,
            "period": self.config.long_short_period,
            "limit": limit,
        }
        return self._get("/futures/data/globalLongShortAccountRatio", params=params)

    def get_funding_rate(self, limit: int = 1) -> List[Dict[str, str]]:
        params = {"symbol": self.config.symbol, "limit": limit}
        return self._get("/fapi/v1/fundingRate", params=params)

    def get_volatility_estimate(self, lookback: int = 24) -> float:
        candles = self.get_historical_candles(limit=lookback)
        close_pct = candles["close"].pct_change().dropna()
        return float(close_pct.std())

    def now(self) -> dt.datetime:
        """Return current UTC time (wrapper kept for convenience/testing)."""

        return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


__all__ = ["BinanceDataFetcher"]
