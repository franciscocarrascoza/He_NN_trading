from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["BinanceDataFetcher", "HermiteDataset", "make_labels"]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy loading
    if name == "BinanceDataFetcher":
        module = import_module("src.data.binance_fetcher")
        return getattr(module, name)
    if name == "HermiteDataset":
        module = import_module("src.data.dataset")
        return getattr(module, name)
    if name == "make_labels":
        module = import_module("src.data.labels")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
