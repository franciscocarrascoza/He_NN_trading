"""Strategy evaluation helpers with gating and Kelly-style sizing."""  # FIX: clarify module purpose

from __future__ import annotations  # FIX: enable postponed evaluation for annotations

from dataclasses import dataclass  # FIX: expose dataclass decorator for metrics containers
from typing import Dict, Sequence

import numpy as np


def _annualised_sharpe(
    returns: np.ndarray,
    seconds_per_step: float,
    *,
    freq_per_year: float | None = None,
) -> float:
    """Compute annualised Sharpe using sample statistics with stability guards."""  # FIX: document Sharpe handling

    if returns.size == 0:  # FIX: handle empty sequences safely
        return float("nan")  # FIX: propagate no-data condition
    mean = float(np.mean(returns))  # FIX: explicit float conversion for clarity
    std = float(np.std(returns, ddof=1))  # FIX: sample std per requirement
    if std <= 0.0:  # FIX: guard against zero dispersion
        return float("nan")  # FIX: avoid divide-by-zero
    periods_per_year = freq_per_year if freq_per_year is not None else (365.0 * 24.0 * 3600.0) / max(  # FIX: allow config-drive
        seconds_per_step,
        1.0,
    )
    return float((mean / std) * np.sqrt(periods_per_year))  # FIX: apply annualisation formula respecting cadence override


def _max_drawdown(returns: np.ndarray) -> float:
    """Return maximum drawdown from a stream of simple returns."""  # FIX: clarify metric domain

    if returns.size == 0:  # FIX: handle empty arrays gracefully
        return float("nan")  # FIX: no drawdown when no trades
    equity = np.cumprod(1.0 + returns)  # FIX: use compounded equity curve
    peak = np.maximum.accumulate(equity)  # FIX: track running peak
    drawdowns = equity / peak - 1.0  # FIX: compute drawdown depth
    return float(drawdowns.min())  # FIX: expose worst drawdown


@dataclass
class StrategyMetrics:
    """Container for per-threshold strategy diagnostics."""  # FIX: describe dataclass intent

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
    """Summary of a threshold sweep including the best Sharpe outcome."""  # FIX: document summary payload

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
    slippage_bps: float = 0.0,
    freq_per_year: float | None = None,
) -> StrategySummary:
    """Evaluate Kelly-sized strategies across multiple thresholds with gating."""  # FIX: update docstring for new params

    returns = np.asarray(returns, dtype=float)  # FIX: ensure numpy array for vector ops
    simple_returns = np.expm1(returns)  # FIX: convert log returns to simple relatives
    probabilities = np.asarray(probabilities, dtype=float)  # FIX: align type for comparisons
    if returns.shape != probabilities.shape:  # FIX: maintain consistent shape assumptions
        raise ValueError("returns and probabilities must share shape.")  # FIX: explicit input validation
    conformal_arr = np.ones_like(probabilities) if conformal_p is None else np.asarray(conformal_p, dtype=float)  # FIX: normalise conformal inputs
    if conformal_arr.shape != probabilities.shape:  # FIX: guard conformal array shape
        raise ValueError("conformal_p must match probabilities shape when provided.")  # FIX: explicit error for mismatch

    thresholds_sorted = sorted(float(t) for t in thresholds)  # FIX: deterministic sweep order
    runs: Dict[float, StrategyMetrics] = {}  # FIX: store per-threshold metrics
    per_trade_cost = (cost_bps + slippage_bps) * 1e-4  # FIX: combine fee and slippage
    conf_margin = max(confidence_margin, 0.0)  # FIX: clamp confidence margin to feasible domain
    clip = max(kelly_clip, 1e-6)  # FIX: avoid zero clip causing flat Kelly

    for threshold in thresholds_sorted:  # FIX: iterate thresholds deterministically
        long_mask = probabilities >= threshold  # FIX: compute threshold-specific long mask
        short_mask = probabilities <= (1.0 - threshold)  # FIX: compute threshold-specific short mask
        base_mask = long_mask | short_mask  # FIX: positions active where either side triggers
        confidence_mask = np.abs(probabilities - 0.5) >= conf_margin  # FIX: enforce confidence gate
        if use_conformal_gate:  # FIX: optionally apply conformal filter
            confidence_mask &= conformal_arr >= conformal_p_min  # FIX: gate on conformal p-value
        active_mask = base_mask & confidence_mask  # FIX: combine structural and confidence gating
        kelly = np.clip(2.0 * (probabilities - 0.5), -clip, clip)  # FIX: maintain Kelly sizing within clip
        positions = np.where(active_mask, kelly, 0.0)  # FIX: zero out inactive positions
        trade_changes = np.diff(np.concatenate(([0.0], positions)))  # FIX: compute per-step position delta
        turnover = float(np.sum(np.abs(trade_changes)))  # FIX: report turnover as sum abs deltas
        gross_returns = positions * simple_returns  # FIX: apply positions to simple returns
        cost = np.abs(trade_changes) * per_trade_cost  # FIX: deduct combined trading frictions
        strategy_returns = gross_returns - cost  # FIX: compute net returns for strategy
        sharpe = _annualised_sharpe(  # FIX: evaluate Sharpe with configurable annualisation frequency
            strategy_returns,
            seconds_per_step,
            freq_per_year=freq_per_year,
        )
        mdd = _max_drawdown(strategy_returns)  # FIX: reuse drawdown logic on net returns
        hit_rate = float(np.mean(strategy_returns > 0.0)) if strategy_returns.size else float("nan")  # FIX: robust hit-rate calc
        active_fraction = float(np.mean(active_mask)) if active_mask.size else float("nan")  # FIX: track time active
        avg_position = float(np.mean(np.abs(positions))) if positions.size else float("nan")  # FIX: mean leverage footprint
        runs[threshold] = StrategyMetrics(
            sharpe=sharpe,
            max_drawdown=mdd,
            hit_rate=hit_rate,
            turnover=turnover,
            returns=strategy_returns.copy(),  # FIX: ensure per-threshold isolation
            threshold=threshold,
            active_fraction=active_fraction,
            avg_position=avg_position,
        )  # FIX: record per-threshold metrics

    best_threshold = thresholds_sorted[0] if thresholds_sorted else 0.5  # FIX: default when no thresholds provided
    best_metrics = runs[best_threshold]  # FIX: initialise with first candidate
    best_score = best_metrics.sharpe if not np.isnan(best_metrics.sharpe) else float("-inf")  # FIX: handle NaN Sharpe
    for thr, metrics in runs.items():  # FIX: evaluate all thresholds for best Sharpe
        sharpe = metrics.sharpe  # FIX: alias for clarity
        score = sharpe if not np.isnan(sharpe) else float("-inf")  # FIX: treat NaN as inferior
        if score > best_score:  # FIX: update best when improved
            best_score = score  # FIX: track best Sharpe
            best_threshold = thr  # FIX: record best threshold
            best_metrics = metrics  # FIX: store best metrics reference

    return StrategySummary(best_threshold=best_threshold, best_metrics=best_metrics, per_threshold=runs)  # FIX: deliver summary


def baseline_strategies(
    returns: np.ndarray,
    *,
    cost_bps: float,
    seconds_per_step: float,
    slippage_bps: float = 0.0,
    freq_per_year: float | None = None,
) -> Dict[str, StrategyMetrics]:
    """Provide simple long/flat baselines under trading frictions."""  # FIX: explain baseline behaviour

    returns = np.asarray(returns, dtype=float)  # FIX: ensure numeric array
    simple_returns = np.expm1(returns)  # FIX: align with strategy return convention
    per_trade_cost = (cost_bps + slippage_bps) * 1e-4  # FIX: mirror cost handling

    always_long_positions = np.ones_like(simple_returns)  # FIX: define long exposure
    trade_changes_long = np.diff(np.concatenate(([0.0], always_long_positions)))  # FIX: compute turnover for long
    cost_long = np.abs(trade_changes_long) * per_trade_cost  # FIX: incorporate trading costs
    long_returns = always_long_positions * simple_returns - cost_long  # FIX: compute net long returns
    long_metrics = StrategyMetrics(
        sharpe=_annualised_sharpe(  # FIX: consistent Sharpe evaluation with optional frequency override
            long_returns,
            seconds_per_step,
            freq_per_year=freq_per_year,
        ),
        max_drawdown=_max_drawdown(long_returns),  # FIX: evaluate drawdown on net curve
        hit_rate=float(np.mean(long_returns > 0.0)) if long_returns.size else float("nan"),  # FIX: guard empty arrays
        turnover=float(np.sum(np.abs(trade_changes_long))),  # FIX: turnover uses sum abs deltas
        returns=long_returns,
    )  # FIX: package long baseline metrics

    flat_returns = np.zeros_like(simple_returns)  # FIX: represent flat strategy
    flat_metrics = StrategyMetrics(
        sharpe=0.0,
        max_drawdown=0.0,
        hit_rate=0.0,
        turnover=0.0,
        returns=flat_returns,
    )  # FIX: neutral baseline metrics

    return {"always_long": long_metrics, "always_flat": flat_metrics}  # FIX: expose baseline dictionary


__all__ = ["StrategyMetrics", "StrategySummary", "baseline_strategies", "evaluate_strategy"]  # FIX: export module API
