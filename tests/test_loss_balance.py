"""Test sign-consistency auxiliary loss and loss balance."""  # FIX: add test for M1 loss balance per spec

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from src.pipeline.training import _sign_consistency_loss


def test_sign_loss_positive_on_mixed_signs() -> None:
    """Synthetic batch where targets have mixed signs; check loss_sign > 0."""  # FIX: test per spec
    targets = torch.tensor([-0.02, 0.03, -0.01, 0.04, 0.01, -0.03], dtype=torch.float32)  # FIX: mixed signs
    mu = torch.tensor([0.01, -0.02, 0.015, -0.01, -0.02, 0.01], dtype=torch.float32)  # FIX: predictions with sign errors

    loss_sign = _sign_consistency_loss(mu, targets)  # FIX: compute sign-consistency loss

    assert loss_sign > 0.0  # FIX: loss must be positive when signs don't match perfectly per spec
    assert torch.isfinite(loss_sign)  # FIX: ensure no NaN or inf


def test_loss_decreases_across_gradient_steps() -> None:
    """Total loss decreases across a few gradient steps on a tiny model."""  # FIX: test per spec
    torch.manual_seed(42)  # FIX: deterministic initialization
    batch_size = 16  # FIX: small batch
    input_dim = 8  # FIX: tiny input

    model = nn.Sequential(  # FIX: minimal model for test
        nn.Linear(input_dim, 16),  # FIX: hidden layer
        nn.ReLU(),  # FIX: nonlinearity
        nn.Linear(16, 1),  # FIX: output layer (mu prediction)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # FIX: optimizer with reasonable LR

    x = torch.randn(batch_size, input_dim)  # FIX: random input
    targets = torch.randn(batch_size, 1) * 0.02  # FIX: small target magnitudes
    reg_weight = 1.0  # FIX: regression weight per spec
    sign_hinge_weight = 0.03  # FIX: sign weight per spec

    losses = []  # FIX: track loss progression
    for step in range(5):  # FIX: few gradient steps per spec
        optimizer.zero_grad()  # FIX: clear gradients
        mu = model(x)  # FIX: forward pass
        nll_loss = ((targets - mu) ** 2).mean()  # FIX: simple squared error as proxy for NLL
        sign_loss = _sign_consistency_loss(mu, targets)  # FIX: compute sign loss
        total_loss = reg_weight * nll_loss + sign_hinge_weight * sign_loss  # FIX: combine per spec
        total_loss.backward()  # FIX: backprop
        optimizer.step()  # FIX: update weights
        losses.append(total_loss.item())  # FIX: record loss

    assert losses[-1] < losses[0]  # FIX: final loss must be lower than initial per spec
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)  # FIX: no NaN or inf
