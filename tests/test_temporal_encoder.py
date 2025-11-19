from __future__ import annotations

"""Unit tests covering the standalone temporal encoder."""  # FIX: test H1 temporal encoder per spec

import os

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

torch = pytest.importorskip("torch")

from src.models import LSTMTemporalEncoder


def test_temporal_encoder_shape() -> None:
    """Build a dummy tensor (B=8, W=64, F=4), run through encoder, assert output shape (8, lstm_hidden)."""  # FIX: test per spec
    encoder = LSTMTemporalEncoder(input_size=4, hidden_size=16)  # FIX: instantiate with 4 features, 16 hidden
    batch = torch.randn(8, 64, 4)  # FIX: synthetic batch (B=8, W=64, F=4) per spec
    output = encoder(batch)  # FIX: forward pass
    assert output.shape == (8, 16)  # FIX: verify shape (B, hidden_size) per spec
