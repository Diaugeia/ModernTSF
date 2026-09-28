"""Clean-room Time-o1 transformed-label alignment objective and local backbone.

Time-o1 is a model-agnostic training objective rather than a forecasting
architecture. This module therefore supplies a small independent temporal
backbone plus the paper's per-variate SVD basis fitting and Equations (4)--(5)
for use by experiment runners.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        alpha: float = 0.8,
        rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model) < 1:
            raise ValueError("lengths, channels, and dimensions must be positive")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not 0 < rank_ratio <= 1:
            raise ValueError("rank_ratio must be in (0, 1]")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )
        self.skip = nn.Linear(seq_len, pred_len)
        identity = torch.eye(pred_len).expand(enc_in, -1, -1).clone()
        self.register_buffer("projection", identity)
        self.register_buffer("projection_ready", torch.tensor(False))

    @torch.no_grad()
    def fit_projection(self, labels: torch.Tensor) -> torch.Tensor:
        """Fit per-variate right-singular-vector bases from training labels."""
        if labels.ndim != 3 or labels.shape[1:] != (self.pred_len, self.enc_in):
            raise ValueError(
                f"expected labels (N, {self.pred_len}, {self.enc_in}), got {tuple(labels.shape)}"
            )
        centered = labels - labels.mean(dim=1, keepdim=True)
        standardized = centered / torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        bases = []
        for channel in range(self.enc_in):
            _, _, right = torch.linalg.svd(standardized[:, :, channel], full_matrices=True)
            bases.append(right.transpose(0, 1))
        self.projection.copy_(torch.stack(bases))
        self.projection_ready.fill_(True)
        return self.projection

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Project horizon values into descending-significance models._components."""
        if values.ndim != 3 or values.shape[1:] != (self.pred_len, self.enc_in):
            raise ValueError(
                f"expected values (B, {self.pred_len}, {self.enc_in}), got {tuple(values.shape)}"
            )
        return torch.einsum("btc,ctk->bkc", values, self.projection)

    def transformed_alignment_loss(
        self, forecast: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute paper Equation (5) using the retained leading models._components."""
        if not bool(self.projection_ready):
            raise RuntimeError("fit_projection must be called on training labels first")
        forecast_components = self.transform(forecast)
        target_components = self.transform(target)
        retained = max(1, round(self.rank_ratio * self.pred_len))
        transformed = (
            forecast_components[:, :retained] - target_components[:, :retained]
        ).abs().sum()
        temporal = (forecast - target).square().sum()
        return self.alpha * transformed + (1 - self.alpha) * temporal

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected input (B, {self.seq_len}, {self.enc_in}), got {tuple(x_enc.shape)}"
            )
        values = x_enc.transpose(1, 2)
        return (self.temporal(values) + self.skip(values)).transpose(1, 2)


__all__ = ["Model"]
