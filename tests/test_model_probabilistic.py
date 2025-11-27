from __future__ import annotations

"""Tests for the Hermite forecaster probabilistic heads."""

import os
from dataclasses import replace

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

torch = pytest.importorskip("torch")

from src.config import MODEL
from src.models import HermiteForecaster, ModelOutput


def test_probabilistic_head_shapes_and_nll() -> None:
    model_cfg = replace(MODEL, hermite_hidden_dim=16, dropout=0.0)
    window_columns = ("f1", "f2", "f3", "f4")
    model = HermiteForecaster(
        input_dim=8,
        model_config=model_cfg,
        feature_window=2,
        window_feature_columns=window_columns,
    )
    x = torch.randn(4, 8)
    output = model(x)
    assert isinstance(output, ModelOutput)
    mu = output.mu
    logvar = output.logvar
    p_up = output.p_up_cdf
    logits = output.logits
    assert mu.shape == (4, 1)
    assert logvar.shape == (4, 1)
    assert p_up.shape == (4, 1)
    assert logits.shape == (4, 1)
    assert torch.all((p_up >= 0.0) & (p_up <= 1.0))
    assert torch.all((output.p_up_logit >= 0.0) & (output.p_up_logit <= 1.0))

    targets = torch.randn(4, 1)
    nll = 0.5 * ((targets - mu) ** 2 * torch.exp(-logvar) + logvar)
    assert torch.isfinite(nll).all()
