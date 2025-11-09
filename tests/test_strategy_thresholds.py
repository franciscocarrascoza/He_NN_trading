"""Strategy threshold sweep regression tests."""  # FIX: cover threshold gating differences

import numpy as np

from src.eval.strategy import evaluate_strategy


def test_threshold_sweep_produces_distinct_metrics() -> None:
    """Ensure adjacent thresholds yield distinct turnover and Sharpe."""  # FIX: document test intent

    log_returns = np.log1p(np.array([0.01, -0.015, 0.02, -0.005, 0.012, -0.01, 0.03], dtype=float))  # FIX: synthetic log returns
    probabilities = np.array([0.7, 0.6, 0.65, 0.4, 0.72, 0.3, 0.8], dtype=float)  # FIX: calibrated probabilities with varied confidence
    summary = evaluate_strategy(
        log_returns,
        probabilities,
        thresholds=[0.55, 0.60],
        cost_bps=5.0,
        seconds_per_step=3600.0,
        confidence_margin=0.05,
        kelly_clip=0.5,
        conformal_p=np.ones_like(probabilities),
        use_conformal_gate=False,
        conformal_p_min=0.0,
    )
    low_thr = summary.per_threshold[0.55]
    high_thr = summary.per_threshold[0.6]
    assert low_thr.turnover != high_thr.turnover  # FIX: verify trade counts differ
    assert low_thr.sharpe != high_thr.sharpe  # FIX: ensure performance metrics diverge
