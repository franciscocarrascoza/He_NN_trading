from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from src.config import DataConfig, EvaluationConfig


@dataclass
class FoldIndices:
    fold_id: int
    train_idx: torch.Tensor
    calibration_idx: torch.Tensor
    val_idx: torch.Tensor
    scaler_idx: torch.Tensor


class RollingOriginSplitter:
    def __init__(
        self,
        *,
        dataset_length: int,
        data_config: DataConfig,
        evaluation_config: EvaluationConfig,
    ) -> None:
        if dataset_length <= 0:
            raise ValueError("dataset_length must be positive.")
        self.dataset_length = dataset_length
        self.data_config = data_config
        self.evaluation_config = evaluation_config

    def _make_calibration_split(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate calibration slice with enforced minimum size per spec."""  # FIX: add docstring
        cal_fraction = max(0.0, min(0.5, self.evaluation_config.calibration_fraction))  # FIX: clamp fraction
        total = indices.shape[0]  # FIX: total available indices
        if total < 2:  # FIX: require at least 2 samples
            raise ValueError("Not enough indices to allocate calibration slice.")  # FIX: explicit error
        min_calib_size = getattr(self.evaluation_config, "min_calib_size", 256)  # FIX: load min from config with fallback
        desired_min = max(int(total * 0.1), min_calib_size)  # FIX: respect configured minimum calibration size per spec
        cal_size = max(1, int(total * cal_fraction))  # FIX: compute from fraction
        if desired_min < total:  # FIX: enforce minimum when feasible
            cal_size = max(cal_size, desired_min)  # FIX: take max of fraction and minimum
        else:
            cal_size = max(cal_size, max(total // 2, 1))  # FIX: fallback when dataset too small
        cal_size = min(cal_size, total - 1)  # FIX: leave at least 1 for training
        if cal_size < min_calib_size and total >= min_calib_size:  # FIX: enforce hard minimum when dataset permits
            raise ValueError(
                f"Calibration block {cal_size} < min_calib_size {min_calib_size}. "
                f"Increase dataset size or reduce cv_folds."
            )  # FIX: explicit error per spec with actionable guidance
        cal_indices = indices[-cal_size:]  # FIX: take last cal_size indices
        train_indices = indices[:-cal_size]  # FIX: remaining for training
        if train_indices.numel() == 0:  # FIX: guard empty training set
            raise ValueError("Not enough training indices after reserving calibration slice.")  # FIX: explicit error
        return train_indices, cal_indices  # FIX: return disjoint splits

    def _single_split(self) -> List[FoldIndices]:
        n = self.dataset_length
        val_len = max(1, int(n * self.data_config.validation_split))
        train_full_len = n - val_len
        if train_full_len <= 1:
            raise ValueError("Dataset too small relative to validation split.")
        scaler_idx = torch.arange(train_full_len)
        train_indices, cal_indices = self._make_calibration_split(scaler_idx)
        val_indices = torch.arange(train_full_len, n)
        return [
            FoldIndices(
                fold_id=0,
                train_idx=train_indices,
                calibration_idx=cal_indices,
                val_idx=val_indices,
                scaler_idx=scaler_idx,
            )
        ]

    def _rolling_origin(self) -> List[FoldIndices]:
        n = self.dataset_length
        val_block = max(1, self.evaluation_config.val_block)
        folds: List[FoldIndices] = []
        min_train_len = max(
            self.data_config.feature_window * 2,
            int(n * self.data_config.validation_split),
            val_block,
        )
        start = min_train_len
        fold_id = 0
        while start < n and fold_id < self.evaluation_config.cv_folds:
            val_end = min(n, start + val_block)
            scaler_idx = torch.arange(0, start)
            if scaler_idx.numel() <= 1:
                break
            train_indices, cal_indices = self._make_calibration_split(scaler_idx)
            val_indices = torch.arange(start, val_end)
            if val_indices.numel() == 0:
                break
            folds.append(
                FoldIndices(
                    fold_id=fold_id,
                    train_idx=train_indices,
                    calibration_idx=cal_indices,
                    val_idx=val_indices,
                    scaler_idx=scaler_idx,
                )
            )
            fold_id += 1
            start += val_block
        if not folds:
            raise ValueError("Unable to construct rolling-origin folds with provided configuration.")
        return folds

    def split(self, *, use_cv: bool) -> List[FoldIndices]:
        if use_cv:
            return self._rolling_origin()
        return self._single_split()


__all__ = ["FoldIndices", "RollingOriginSplitter"]
