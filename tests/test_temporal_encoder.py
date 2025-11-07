from __future__ import annotations

"""Unit tests covering the standalone temporal encoder."""

import os

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

torch = pytest.importorskip("torch")

from src.models import LSTMTemporalEncoder


def test_temporal_encoder_shape() -> None:
    encoder = LSTMTemporalEncoder(input_size=4, hidden_size=16)
    batch = torch.randn(8, 64, 4)
    output = encoder(batch)
    assert output.shape == (8, 16)
