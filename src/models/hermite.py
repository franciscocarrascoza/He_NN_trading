from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.config import TRAINING, TrainingConfig


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


class HermiteActivation(nn.Module):
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
        else:
            self.register_buffer("coeffs", coeffs)
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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hermite = HermiteActivation(degree=degree)
        self.a_maps = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(maps_a)])
        self.b_maps = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(maps_b)])
        self.bias = nn.Parameter(torch.zeros(input_dim))

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


class HermiteForecaster(nn.Module):
    """Compact neural network that consumes engineered features and forecasts price."""

    def __init__(
        self,
        input_dim: int,
        config: TrainingConfig = TRAINING,
    ) -> None:
        super().__init__()
        self.block = SymmetricHermiteBlock(
            input_dim=input_dim,
            hidden_dim=config.hermite_hidden_dim,
            degree=config.hermite_degree,
            maps_a=config.hermite_maps_a,
            maps_b=config.hermite_maps_b,
        )
        feature_dim = input_dim + input_dim + 1
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, config.hermite_hidden_dim),
            nn.LayerNorm(config.hermite_hidden_dim),
            nn.GELU(),
            nn.Linear(config.hermite_hidden_dim, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(feature_dim, config.hermite_hidden_dim),
            nn.LayerNorm(config.hermite_hidden_dim),
            nn.GELU(),
            nn.Linear(config.hermite_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        symmetric_features, jacobian_trace = self.block(x)
        features = torch.cat([x, symmetric_features, jacobian_trace], dim=-1)
        pred_log_return = self.regression_head(features)
        direction_logits = self.direction_head(features)
        return pred_log_return, direction_logits


__all__ = [
    "HermiteActivation",
    "SymmetricHermiteBlock",
    "HermiteForecaster",
]
