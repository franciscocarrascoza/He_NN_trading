from __future__ import annotations

"""Strategy evaluation helpers with gating and Kelly-style sizing."""

from dataclasses import dataclass
from typing import Dict, Sequence

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
    threshold: float | None = None
    active_fraction: float | None = None
    avg_position: float | None = None


@dataclass
class StrategySummary:
    best_threshold: float
    best_metrics: StrategyMetrics
    per_threshold: Dict[float, StrategyMetrics]


def evaluate_strategy(
    returns: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Sequence[float],
    cost_bps: float,
    seconds_per_step: float,
    confidence_margin: float,
    kelly_clip: float,
    conformal_p: np.ndarray | None = None,
    use_conformal_gate: bool = True,
    conformal_p_min: float = 0.05,
) -> StrategySummary:
    """Evaluate Kelly-sized strategies across multiple thresholds and gate filters."""

    returns = np.asarray(returns, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if returns.shape != probabilities.shape:
        raise ValueError("returns and probabilities must share shape.")
    conformal_arr = np.ones_like(probabilities) if conformal_p is None else np.asarray(conformal_p, dtype=float)
    if conformal_arr.shape != probabilities.shape:
        raise ValueError("conformal_p must match probabilities shape when provided.")

    thresholds_sorted = sorted(float(t) for t in thresholds)
    runs: Dict[float, StrategyMetrics] = {}
    cost_per_trade = cost_bps * 1e-4
    conf_margin = max(confidence_margin, 0.0)
    clip = max(kelly_clip, 1e-6)

    for threshold in thresholds_sorted:
        long_mask = probabilities >= threshold
        short_mask = probabilities <= (1.0 - threshold)
        base_mask = long_mask | short_mask
        confidence_mask = np.abs(probabilities - 0.5) >= conf_margin
        if use_conformal_gate:
            confidence_mask &= conformal_arr >= conformal_p_min
        active_mask = base_mask & confidence_mask
        kelly = np.clip(2.0 * (probabilities - 0.5), -clip, clip)
        positions = np.where(active_mask, kelly, 0.0)
        trade_changes = np.diff(np.concatenate([[0.0], positions]))
        turnover = float(np.mean(np.abs(trade_changes)))
        cost = np.abs(trade_changes) * cost_per_trade
        strategy_returns = positions * returns - cost
        sharpe = _annualised_sharpe(strategy_returns, seconds_per_step)
        mdd = _max_drawdown(strategy_returns)
        hit_rate = float(np.mean(strategy_returns > 0.0)) if strategy_returns.size else float("nan")
        active_fraction = float(np.mean(active_mask)) if active_mask.size else float("nan")
        avg_position = float(np.mean(np.abs(positions))) if positions.size else float("nan")
        runs[threshold] = StrategyMetrics(
            sharpe=sharpe,
            max_drawdown=mdd,
            hit_rate=hit_rate,
            turnover=turnover,
            returns=strategy_returns,
            threshold=threshold,
            active_fraction=active_fraction,
            avg_position=avg_position,
        )

    best_threshold = thresholds_sorted[0] if thresholds_sorted else 0.5
    best_metrics = runs[best_threshold]
    best_score = best_metrics.sharpe if not np.isnan(best_metrics.sharpe) else float("-inf")
    for thr, metrics in runs.items():
        sharpe = metrics.sharpe
        score = sharpe if not np.isnan(sharpe) else float("-inf")
        if score > best_score:
            best_score = score
            best_threshold = thr
            best_metrics = metrics

    return StrategySummary(best_threshold=best_threshold, best_metrics=best_metrics, per_threshold=runs)


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


__all__ = ["StrategyMetrics", "StrategySummary", "baseline_strategies", "evaluate_strategy"]
