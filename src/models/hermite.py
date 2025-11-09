from __future__ import annotations

"""Hermite-based probabilistic forecaster with optional temporal encoder."""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import MODEL, ModelConfig

SQRT_TWO = float(np.sqrt(2.0))


def hermite_polynomials(z: torch.Tensor, degree: int, *, physicist: bool = False) -> torch.Tensor:
    polys = [torch.ones_like(z)]
    if degree >= 1:
        polys.append(z if not physicist else 2.0 * z)
    for n in range(1, degree):
        if physicist:
            next_poly = 2.0 * z * polys[n] - 2.0 * n * polys[n - 1]
        else:
            next_poly = z * polys[n] - n * polys[n - 1]
        polys.append(next_poly)
    return torch.stack(polys, dim=0)


def _init_positive_small(param: torch.nn.Parameter) -> None:
    with torch.no_grad():
        param.copy_(param.abs() * 0.1)


@dataclass
class ModelOutput:
    """Structured output for HermiteForecaster forward pass."""

    mu: torch.Tensor
    logvar: torch.Tensor
    logits: torch.Tensor
    p_up_cdf: torch.Tensor
    p_up_logit: torch.Tensor

    def probability(self, source: str) -> torch.Tensor:
        if source.lower() == "cdf":
            return self.p_up_cdf
        return self.p_up_logit


class HermiteActivation(nn.Module):
    """Truncated Hermite polynomial activation with learnable coefficients."""

    def __init__(
        self,
        degree: int,
        learnable_coeffs: bool = True,
        version: str = "probabilist",
    ) -> None:
        super().__init__()
        self.degree = degree
        self.version = version
        coeffs = torch.randn(degree + 1) * 0.1
        if learnable_coeffs:
            self.coeffs = nn.Parameter(coeffs)
            _init_positive_small(self.coeffs)
        else:
            self.register_buffer("coeffs", coeffs.abs() * 0.1)
        self.register_buffer("d0", torch.tensor(0.0))
        self.register_buffer("d1", torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.clamp(z, min=-10.0, max=10.0)
        polys = hermite_polynomials(z, self.degree, physicist=self.version == "physicist")
        coeffs = self.coeffs.view(self.degree + 1, *([1] * z.ndim))
        return (polys * coeffs).sum(dim=0) + self.d0 + self.d1 * z

    def derivative(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.clamp(z, min=-10.0, max=10.0)
        if self.degree == 0:
            return torch.zeros_like(z)
        polys = hermite_polynomials(z, self.degree - 1, physicist=self.version == "physicist")
        if self.version == "physicist":
            factors = torch.arange(1, self.degree + 1, device=z.device, dtype=z.dtype) * 2
        else:
            factors = torch.arange(1, self.degree + 1, device=z.device, dtype=z.dtype)
        coeffs = self.coeffs[1:].view(self.degree, *([1] * z.ndim))
        factors = factors.view(self.degree, *([1] * z.ndim))
        return (coeffs * factors * polys).sum(dim=0) + self.d1


class SymmetricHermiteBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        degree: int,
        maps_a: int,
        maps_b: int,
        activation_version: str,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hermite = HermiteActivation(degree=degree, version=activation_version)
        self.a_maps = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(maps_a)])
        self.b_maps = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(maps_b)])
        self.bias = nn.Parameter(torch.zeros(input_dim))
        for linear in list(self.a_maps) + list(self.b_maps):
            _init_positive_small(linear.weight)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = s.size(0)
        aggregate = torch.zeros(batch, self.input_dim, device=s.device, dtype=s.dtype)
        jacobian_trace = torch.zeros(batch, 1, device=s.device, dtype=s.dtype)

        for linear in self.a_maps:
            z = linear(s)
            h = self.hermite(z)
            aggregate = aggregate + torch.matmul(h, linear.weight)
            deriv = self.hermite.derivative(z)
            row_norm_sq = (linear.weight.pow(2)).sum(dim=1)
            jacobian_trace = jacobian_trace + torch.matmul(deriv, row_norm_sq.unsqueeze(-1))

        for linear in self.b_maps:
            z = linear(s)
            h = self.hermite(z)
            aggregate = aggregate - torch.matmul(h, linear.weight)
            deriv = self.hermite.derivative(z)
            row_norm_sq = (linear.weight.pow(2)).sum(dim=1)
            jacobian_trace = jacobian_trace - torch.matmul(deriv, row_norm_sq.unsqueeze(-1))

        return aggregate + self.bias, jacobian_trace


class LSTMTemporalEncoder(nn.Module):
    """Lightweight LSTM encoder for windowed features."""

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive for LSTMTemporalEncoder.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive for LSTMTemporalEncoder.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences and return the last hidden state."""

        if sequences.ndim != 3:
            raise ValueError("LSTMTemporalEncoder expects input of shape (B, T, F).")
        _, (hidden, _) = self.lstm(sequences)
        hidden = hidden.view(self.num_layers, self.num_directions, sequences.size(0), self.hidden_size)
        last_layer_hidden = hidden[-1]
        last_hidden = last_layer_hidden.transpose(0, 1).reshape(sequences.size(0), -1)
        return last_hidden


class HermiteForecaster(nn.Module):
    """Probabilistic Hermite forecaster with stabilised variance and classification heads."""  # FIX: highlight variance clamp

    def __init__(
        self,
        input_dim: int,
        *,
        model_config: ModelConfig = MODEL,
        feature_window: int | None = None,
        window_feature_columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.use_lstm = getattr(model_config, "use_lstm", False)
        self.feature_window = feature_window
        self.window_feature_columns = tuple(window_feature_columns) if window_feature_columns else None
        self.temporal_encoder: LSTMTemporalEncoder | None = None
        self.seq_flat_dim = 0
        self.feats_per_step = 0
        context_dim = input_dim
        if self.use_lstm:
            if feature_window is None or feature_window <= 0:
                raise ValueError("feature_window must be provided and positive when use_lstm is True.")
            if not window_feature_columns:
                raise ValueError("window_feature_columns must be provided when use_lstm is True.")
            self.feats_per_step = len(window_feature_columns)
            if self.feats_per_step == 0:
                raise ValueError("window_feature_columns cannot be empty when use_lstm is True.")
            self.seq_flat_dim = feature_window * self.feats_per_step
            if self.seq_flat_dim > input_dim:
                raise ValueError("Sequence feature dimension exceeds provided input_dim.")
            context_dim = input_dim - self.seq_flat_dim
            self.temporal_encoder = LSTMTemporalEncoder(
                input_size=self.feats_per_step,
                hidden_size=model_config.lstm_hidden,
            )
            block_input_dim = context_dim + self.temporal_encoder.output_dim
        else:
            block_input_dim = input_dim
        if block_input_dim <= 0:
            raise ValueError("Derived block input dimension must be positive.")
        self.block_input_dim = block_input_dim
        self.block = SymmetricHermiteBlock(
            input_dim=block_input_dim,
            hidden_dim=model_config.hermite_hidden_dim,
            degree=model_config.hermite_degree,
            maps_a=model_config.hermite_maps_a,
            maps_b=model_config.hermite_maps_b,
            activation_version=model_config.hermite_version,
        )
        feature_dim = block_input_dim + block_input_dim + 1
        hidden = model_config.hermite_hidden_dim
        dropout = model_config.dropout

        self.pre_head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        def _make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        self.mu_head = _make_head()
        self.logvar_head = _make_head()
        self.logit_head = _make_head()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if self.use_lstm:
            if x.shape[1] < self.seq_flat_dim:
                raise ValueError("Input feature dimension is smaller than the expected sequence length.")
            seq_flat = x[:, : self.seq_flat_dim]
            seq = seq_flat.view(x.size(0), self.feature_window, self.feats_per_step)
            context = x[:, self.seq_flat_dim :]
            if self.temporal_encoder is None or self.feature_window is None:
                raise RuntimeError("Temporal encoder is not initialised while use_lstm is True.")
            encoded = self.temporal_encoder(seq)
            if context.shape[1] == 0:
                block_input = encoded
            else:
                block_input = torch.cat([encoded, context], dim=-1)
        else:
            block_input = x
        symmetric_features, jacobian_trace = self.block(block_input)
        features = torch.cat([block_input, symmetric_features, jacobian_trace], dim=-1)
        shared = self.pre_head(features)
        mu = self.mu_head(shared)
        logvar = self.logvar_head(shared).clamp(min=-10.0, max=5.0)  # FIX: tame log-variance extremes
        logits = self.logit_head(shared)
        variance = logvar.exp().clamp_min(1e-6)  # FIX: stabilise variance floor
        sigma = variance.sqrt()  # FIX: compute standard deviation post clamp
        denom = torch.clamp(sigma * SQRT_TWO, min=1e-6)
        p_up_cdf = 0.5 * (1.0 + torch.erf(mu / denom))
        p_up_logit = torch.sigmoid(logits)
        return ModelOutput(mu=mu, logvar=logvar, logits=logits, p_up_cdf=p_up_cdf, p_up_logit=p_up_logit)


__all__ = [
    "HermiteActivation",
    "ModelOutput",
    "LSTMTemporalEncoder",
    "SymmetricHermiteBlock",
    "HermiteForecaster",
]
