"""Strategy confidence gating tests."""  # FIX: ensure low-confidence trades are filtered

import numpy as np

from src.eval.strategy import evaluate_strategy


def test_confidence_margin_blocks_low_probability_trades() -> None:
    """Probabilities within the confidence margin should not trigger trades."""  # FIX: describe gating expectation

    log_returns = np.log1p(np.array([0.02, -0.01, 0.015], dtype=float))  # FIX: sample log returns
    probabilities = np.array([0.56, 0.72, 0.40], dtype=float)  # FIX: first entry below confidence margin
    summary = evaluate_strategy(
        log_returns,
        probabilities,
        thresholds=[0.55],
        cost_bps=0.0,
        seconds_per_step=3600.0,
        confidence_margin=0.10,
        kelly_clip=0.5,
        conformal_p=np.ones_like(probabilities),
        use_conformal_gate=False,
        conformal_p_min=0.0,
    )
    metrics = summary.per_threshold[0.55]
    assert metrics.returns[0] == 0.0  # FIX: low-confidence observation is ignored
    assert metrics.active_fraction < 1.0  # FIX: not all samples should be active
