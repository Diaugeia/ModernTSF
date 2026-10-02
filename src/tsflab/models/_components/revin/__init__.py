"""Reversible per-instance normalization for batch-time-channel tensors."""

from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Normalize one sequence instance and later restore its original scale."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if num_features < 1 or eps <= 0:
            raise ValueError("num_features and eps must be positive")
        self.num_features = num_features
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        self.enabled = bool(enabled)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._center: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None

    def forward(self, values: torch.Tensor, mode: str) -> torch.Tensor:
        if not self.enabled:
            return values
        if values.ndim < 3 or values.shape[-1] != self.num_features:
            raise ValueError(
                f"expected final feature width {self.num_features}, got {tuple(values.shape)}"
            )
        if mode == "norm":
            axes = tuple(range(1, values.ndim - 1))
            mean = values.mean(dim=axes, keepdim=True)
            center = (
                values.select(1, values.shape[1] - 1).unsqueeze(1)
                if self.subtract_last
                else mean
            )
            scale = (
                values.var(dim=axes, keepdim=True, unbiased=False)
                .add(self.eps)
                .sqrt()
            )
            self._center, self._scale = center.detach(), scale.detach()
            normalized = (values - self._center) / self._scale
            if self.affine:
                normalized = normalized * self.affine_weight + self.affine_bias
            return normalized
        if mode == "denorm":
            if self._center is None or self._scale is None:
                raise RuntimeError("denorm requires a preceding norm call")
            restored = values
            if self.affine:
                restored = (restored - self.affine_bias) / (
                    self.affine_weight + self.eps * self.eps
                )
            return restored * self._scale + self._center
        raise ValueError("mode must be 'norm' or 'denorm'")
