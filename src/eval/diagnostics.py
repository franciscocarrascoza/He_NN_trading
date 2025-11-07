from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from src.utils.utils import pt_test


def _normal_sf(value: float) -> float:
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def binomial_test_pvalue(successes: int, trials: int, p: float = 0.5) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials inclusive.")
    if trials == 0:
        return float("nan")
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    probabilities = []
    for k in range(trials + 1):
        prob = math.comb(trials, k) * (p ** k) * ((1 - p) ** (trials - k))
        probabilities.append(prob)
    observed = probabilities[successes]
    tail = sum(prob for prob in probabilities if prob <= observed + 1e-15)
    return min(1.0, max(0.0, tail))


@dataclass
class DieboldMarianoResult:
    p_value: float
    mean_loss_diff: float


def diebold_mariano(
    y_true: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    *,
    horizon: int,
    loss: str = "mse",
) -> DieboldMarianoResult:
    y_true = np.asarray(y_true, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    if y_true.shape != forecast.shape or y_true.shape != benchmark.shape:
        raise ValueError("Inputs to diebold_mariano must share shapes.")
    if y_true.size < 2:
        return DieboldMarianoResult(p_value=float("nan"), mean_loss_diff=float("nan"))

    if loss == "mse":
        loss_model = (y_true - forecast) ** 2
        loss_bench = (y_true - benchmark) ** 2
    elif loss == "mae":
        loss_model = np.abs(y_true - forecast)
        loss_bench = np.abs(y_true - benchmark)
    else:
        raise ValueError("loss must be 'mse' or 'mae'.")

    d = loss_model - loss_bench
    d_mean = d.mean()
    d_centered = d - d_mean
    n = d.size
    if horizon < 1:
        horizon = 1
    lag = min(horizon, n - 1)
    gamma = np.empty(lag + 1, dtype=float)
    gamma[0] = np.dot(d_centered, d_centered) / n
    for k in range(1, lag + 1):
        gamma[k] = np.dot(d_centered[k:], d_centered[:-k]) / n
    var_term = gamma[0] + 2.0 * sum((1.0 - k / n) * gamma[k] for k in range(1, lag + 1))
    if var_term <= 0:
        var_term = float(np.var(d_centered, ddof=1))
        if var_term <= 0:
            return DieboldMarianoResult(p_value=float("nan"), mean_loss_diff=float("nan"))
    dm_stat = d_mean / math.sqrt(var_term / n)
    p_value = 2.0 * _normal_sf(abs(dm_stat))
    bounded = min(1.0, max(0.0, p_value))
    return DieboldMarianoResult(p_value=bounded, mean_loss_diff=float(d_mean))


@dataclass
class MincerZarnowitzResult:
    intercept: float
    slope: float
    f_stat: float
    p_value: float


def mincer_zarnowitz(y_true: np.ndarray, forecast: np.ndarray) -> MincerZarnowitzResult:
    y_true = np.asarray(y_true, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if y_true.shape != forecast.shape:
        raise ValueError("y_true and forecast must share shapes.")
    n = y_true.size
    if n < 3:
        return MincerZarnowitzResult(intercept=float("nan"), slope=float("nan"), f_stat=float("nan"), p_value=float("nan"))

    X = np.column_stack([np.ones(n), forecast])
    beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    predictions = X @ beta
    residuals = y_true - predictions
    sse_u = float(np.sum(residuals**2))

    restricted_residuals = y_true - forecast
    sse_r = float(np.sum(restricted_residuals**2))
    q = 2  # restrictions: intercept = 0, slope = 1
    df_u = n - X.shape[1]
    if df_u <= 0 or sse_u <= 0:
        return MincerZarnowitzResult(intercept=beta[0], slope=beta[1], f_stat=float("nan"), p_value=float("nan"))

    numerator = (sse_r - sse_u) / q
    denominator = sse_u / df_u
    if denominator <= 0:
        return MincerZarnowitzResult(intercept=beta[0], slope=beta[1], f_stat=float("nan"), p_value=float("nan"))
    f_stat = max(0.0, numerator / denominator)
    chi2 = q * f_stat
    p_value = math.exp(-chi2 / 2.0)
    return MincerZarnowitzResult(intercept=beta[0], slope=beta[1], f_stat=f_stat, p_value=p_value)


def runs_test(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    signs = np.sign(residuals)
    signs = signs[signs != 0]
    if signs.size < 20:
        return float("nan")
    runs = 1
    for prev, curr in zip(signs[:-1], signs[1:]):
        if curr != prev:
            runs += 1
    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    n = n_pos + n_neg
    expected_runs = ((2 * n_pos * n_neg) / n) + 1
    variance_runs = (
        (2 * n_pos * n_neg) * (2 * n_pos * n_neg - n_pos - n_neg)
        / ((n**2) * (n - 1))
    )
    if variance_runs <= 0:
        return float("nan")
    z_score = (runs - expected_runs) / math.sqrt(variance_runs)
    p_value = 2.0 * _normal_sf(abs(z_score))
    return min(1.0, max(0.0, p_value))


def _chi2_sf_even(x: float, df: int) -> float:
    if df % 2 != 0:
        raise ValueError("Degrees of freedom must be even for _chi2_sf_even.")
    k = df // 2
    term = math.exp(-x / 2.0)
    cumulative = 0.0
    for i in range(k):
        cumulative += (x / 2.0) ** i / math.factorial(i)
    cdf = 1.0 - term * cumulative
    return max(0.0, min(1.0, 1.0 - cdf))


def ljung_box(residuals: np.ndarray, *, max_lag: Optional[int] = None) -> Tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals - residuals.mean()
    n = residuals.size
    if n < 20:
        return float("nan"), float("nan")
    if max_lag is None:
        max_lag = min(10, n // 4)
    max_lag = max(1, max_lag)
    if max_lag % 2 == 1:
        max_lag -= 1
    if max_lag <= 0:
        max_lag = 2
    denom = np.dot(residuals, residuals)
    rho = []
    for lag in range(1, max_lag + 1):
        cov = np.dot(residuals[lag:], residuals[:-lag]) / (n - lag)
        rho.append(cov / (denom / n))
    rho = np.array(rho, dtype=float)
    Q = n * (n + 2) * np.sum((rho**2) / (n - np.arange(1, max_lag + 1)))
    p_value = _chi2_sf_even(Q, max_lag)
    return float(Q), float(p_value)


@dataclass
class CalibrationMetrics:
    brier: float
    auc: float
    ece: float
    mce: float
    brier_uncertainty: float
    brier_resolution: float
    brier_reliability: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_counts: np.ndarray


def _binary_auc(prob: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = prob.argsort().argsort().astype(float) + 1.0
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def probability_calibration_metrics(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationMetrics:
    y_true = np.asarray(y_true, dtype=float)
    prob_up = np.asarray(prob_up, dtype=float)
    if y_true.shape != prob_up.shape:
        raise ValueError("Shapes of y_true and prob_up must match.")
    labels = (y_true > 0).astype(int)
    brier = float(np.mean((prob_up - labels) ** 2))
    auc = _binary_auc(prob_up, labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(prob_up, bin_edges, right=True) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    bin_confidence = np.zeros(n_bins, dtype=float)
    bin_accuracy = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for bin_idx in range(n_bins):
        mask = indices == bin_idx
        count = mask.sum()
        bin_counts[bin_idx] = count
        if count == 0:
            continue
        bin_confidence[bin_idx] = prob_up[mask].mean()
        bin_accuracy[bin_idx] = labels[mask].mean()

    ece = 0.0
    total = prob_up.size
    base_rate = float(labels.mean()) if total > 0 else float("nan")
    reliability_component = 0.0
    resolution_component = 0.0
    max_calibration_error = 0.0
    for idx in range(n_bins):
        if bin_counts[idx] == 0:
            continue
        weight = bin_counts[idx] / total
        diff = bin_accuracy[idx] - bin_confidence[idx]
        ece += weight * abs(diff)
        reliability_component += weight * (diff ** 2)
        resolution_component += weight * ((bin_accuracy[idx] - base_rate) ** 2)
        max_calibration_error = max(max_calibration_error, abs(diff))
    uncertainty_component = base_rate * (1.0 - base_rate) if not math.isnan(base_rate) else float("nan")

    return CalibrationMetrics(
        brier=brier,
        auc=auc,
        ece=float(ece),
        mce=float(max_calibration_error),
        brier_uncertainty=float(uncertainty_component),
        brier_resolution=float(resolution_component),
        brier_reliability=float(reliability_component),
        bin_edges=bin_edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_counts=bin_counts,
    )


def pesaran_timmermann(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Wrapper around pt_test for compatibility with diagnostics."""

    return pt_test(actual, predicted)


__all__ = [
    "CalibrationMetrics",
    "DieboldMarianoResult",
    "MincerZarnowitzResult",
    "binomial_test_pvalue",
    "diebold_mariano",
    "ljung_box",
    "mincer_zarnowitz",
    "pesaran_timmermann",
    "probability_calibration_metrics",
    "runs_test",
]
