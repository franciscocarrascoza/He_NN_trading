from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _annualised_sharpe(returns: np.ndarray, seconds_per_step: float) -> float:
    if returns.size == 0:
        return float("nan")
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std <= 0:
        return float("nan")
    periods_per_year = (365.0 * 24.0 * 3600.0) / max(seconds_per_step, 1.0)
    return float((mean / std) * np.sqrt(periods_per_year))


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / peak - 1.0
    return float(drawdowns.min())


@dataclass
class StrategyMetrics:
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    returns: np.ndarray


def evaluate_strategy(
    returns: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    cost_bps: float,
    seconds_per_step: float,
) -> StrategyMetrics:
    returns = np.asarray(returns, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if returns.shape != probabilities.shape:
        raise ValueError("returns and probabilities must share shape.")

    long_mask = probabilities >= threshold
    short_mask = probabilities <= (1.0 - threshold)
    positions = np.zeros_like(returns)
    positions[long_mask] = 1.0
    positions[short_mask] = -1.0

    trade_changes = np.diff(np.concatenate([[0.0], positions]))
    turnover = float(np.mean(np.abs(trade_changes)))

    cost = np.abs(trade_changes) * (cost_bps * 1e-4)
    strategy_returns = positions * returns - cost

    sharpe = _annualised_sharpe(strategy_returns, seconds_per_step)
    mdd = _max_drawdown(strategy_returns)
    hit_rate = float(np.mean(strategy_returns > 0.0)) if strategy_returns.size else float("nan")

    return StrategyMetrics(
        sharpe=sharpe,
        max_drawdown=mdd,
        hit_rate=hit_rate,
        turnover=turnover,
        returns=strategy_returns,
    )


def baseline_strategies(
    returns: np.ndarray,
    *,
    cost_bps: float,
    seconds_per_step: float,
) -> Dict[str, StrategyMetrics]:
    returns = np.asarray(returns, dtype=float)
    cost_per_trade = cost_bps * 1e-4

    always_long_positions = np.ones_like(returns)
    trade_changes_long = np.diff(np.concatenate([[0.0], always_long_positions]))
    cost_long = np.abs(trade_changes_long) * cost_per_trade
    long_returns = always_long_positions * returns - cost_long
    long_metrics = StrategyMetrics(
        sharpe=_annualised_sharpe(long_returns, seconds_per_step),
        max_drawdown=_max_drawdown(long_returns),
        hit_rate=float(np.mean(long_returns > 0.0)) if long_returns.size else float("nan"),
        turnover=float(np.mean(np.abs(trade_changes_long))),
        returns=long_returns,
    )

    flat_returns = np.zeros_like(returns)
    flat_metrics = StrategyMetrics(
        sharpe=0.0,
        max_drawdown=0.0,
        hit_rate=0.0,
        turnover=0.0,
        returns=flat_returns,
    )

    return {"always_long": long_metrics, "always_flat": flat_metrics}


__all__ = ["StrategyMetrics", "baseline_strategies", "evaluate_strategy"]
