from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import hashlib
import torch


@dataclass
class ScalerStats:
    mean: torch.Tensor
    std: torch.Tensor
    index_hash: str


class LeakageGuardScaler:
    """Simple standard scaler that guards against future-index leakage."""

    def __init__(self, *, max_index: int) -> None:
        self.max_index = max_index
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._hash: Optional[str] = None

    def fit(self, features: torch.Tensor, indices: torch.Tensor) -> ScalerStats:
        if features.ndim != 2:
            raise ValueError("features must be a 2-D tensor.")
        if torch.any(indices > self.max_index):
            raise ValueError("Leakage guard triggered: indices exceed allowed training range.")
        sorted_indices = torch.sort(indices.flatten().to(torch.int64)).values
        index_bytes = sorted_indices.cpu().numpy().tobytes()
        digest = hashlib.sha256(index_bytes).hexdigest()
        subset = features.index_select(0, sorted_indices)
        mean = subset.mean(dim=0)
        std = subset.std(dim=0, unbiased=False).clamp_min(1e-6)
        self._mean = mean
        self._std = std
        self._hash = digest
        return ScalerStats(mean=mean, std=std, index_hash=digest)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        return (features - self._mean) / self._std

    def fit_transform(self, features: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, ScalerStats]:
        stats = self.fit(features, indices)
        return self.transform(features), stats

    @property
    def stats(self) -> ScalerStats:
        if self._mean is None or self._std is None or self._hash is None:
            raise RuntimeError("Scaler not fitted.")
        return ScalerStats(mean=self._mean, std=self._std, index_hash=self._hash)
