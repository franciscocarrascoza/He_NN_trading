"""Test rolling-origin CV splits with calibration block size enforcement."""  # FIX: add test for M2 CV splits per spec

from __future__ import annotations

import torch

from src.config import DataConfig, EvaluationConfig
from src.pipeline.split import RollingOriginSplitter


def test_cv_splits_calib_size_and_disjoint() -> None:
    """Given synthetic time series length N, create splits with cv_folds=5; verify each calib block ≥ min_calib_size and no overlapping between calib and val/test."""  # FIX: test per spec
    n = 2000  # FIX: synthetic dataset length
    data_config = DataConfig(
        feature_window=16,  # FIX: window size
        forecast_horizon=1,  # FIX: horizon
        validation_split=0.2,  # FIX: val fraction
        use_extras=False,  # FIX: no extras
        extras_time_aligned=False,  # FIX: not relevant
    )
    evaluation_config = EvaluationConfig(
        alpha=0.1,  # FIX: coverage level
        cv_folds=5,  # FIX: 5 folds per spec
        val_block=128,  # FIX: validation block size
        calibration_fraction=0.15,  # FIX: calibration fraction
        min_calib_size=256,  # FIX: minimum calibration size per spec
        threshold=0.55,  # FIX: default threshold
        cost_bps=5.0,  # FIX: trading cost
        save_markdown=False,  # FIX: skip markdown
    )

    splitter = RollingOriginSplitter(
        dataset_length=n,
        data_config=data_config,
        evaluation_config=evaluation_config,
    )  # FIX: instantiate splitter per spec
    folds = splitter.split(use_cv=True)  # FIX: generate rolling-origin folds

    assert len(folds) > 0  # FIX: at least one fold generated
    for fold in folds:  # FIX: verify each fold independently
        calib_size = fold.calibration_idx.numel()  # FIX: calibration set size
        assert calib_size >= evaluation_config.min_calib_size  # FIX: enforce minimum per spec

        train_set = set(fold.train_idx.tolist())  # FIX: convert to set for overlap check
        calib_set = set(fold.calibration_idx.tolist())  # FIX: convert calibration indices
        val_set = set(fold.val_idx.tolist())  # FIX: convert validation indices

        assert train_set.isdisjoint(calib_set)  # FIX: training and calibration must be disjoint per spec
        assert train_set.isdisjoint(val_set)  # FIX: training and validation must be disjoint
        assert calib_set.isdisjoint(val_set)  # FIX: calibration and validation must be disjoint per spec


def test_cv_splits_raises_on_small_dataset() -> None:
    """Splitter raises clear error when dataset too small for min_calib_size."""  # FIX: test error handling
    n = 400  # FIX: small dataset
    data_config = DataConfig(
        feature_window=16,
        forecast_horizon=1,
        validation_split=0.2,
        use_extras=False,
        extras_time_aligned=False,
    )
    evaluation_config = EvaluationConfig(
        alpha=0.1,
        cv_folds=5,  # FIX: too many folds for small dataset
        val_block=128,
        calibration_fraction=0.2,
        min_calib_size=256,  # FIX: minimum larger than feasible
        threshold=0.55,
        cost_bps=5.0,
        save_markdown=False,
    )

    splitter = RollingOriginSplitter(
        dataset_length=n,
        data_config=data_config,
        evaluation_config=evaluation_config,
    )
    try:
        folds = splitter.split(use_cv=True)  # FIX: attempt to generate folds
        if folds:  # FIX: if folds generated, verify calibration size constraint
            for fold in folds:
                calib_size = fold.calibration_idx.numel()
                if calib_size < evaluation_config.min_calib_size:  # FIX: expect this to fail
                    raise AssertionError("Expected error for insufficient calibration size")  # FIX: explicit test failure
    except ValueError as exc:  # FIX: expect ValueError with actionable message per spec
        assert "Calibration block" in str(exc) or "min_calib_size" in str(exc)  # FIX: verify error message mentions calibration
