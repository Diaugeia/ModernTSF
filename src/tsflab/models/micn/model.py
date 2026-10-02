"""Clean-room MICN forecast implementation from the ICLR paper.

Each multi-scale branch uses a strided convolution for local patterns, an
isometric convolution over the downsampled sequence for global context, and a
transposed convolution to restore the original temporal grid.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.series_decomposition import SeriesDecomposition


class MultiScaleDecomposition(nn.Module):
    """Average several moving-average trends as described by MICN."""
    def __init__(self, kernels):
        super().__init__()
        self.decompositions = nn.ModuleList([SeriesDecomposition(k) for k in kernels])

    def forward(self, values):
        pairs = [decomposition(values) for decomposition in self.decompositions]
        seasonal = torch.stack([pair[0] for pair in pairs]).mean(0)
        trend = torch.stack([pair[1] for pair in pairs]).mean(0)
        return seasonal, trend


class IsometricConvolutionBranch(nn.Module):
    """Downsample -> global isometric convolution -> upsample at one scale."""
    def __init__(self, width: int, scale: int, dropout: float):
        super().__init__()
        self.scale = scale
        self.local = nn.Conv1d(width, width, scale, stride=scale, groups=width)
        self.global_context = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, groups=width),
            nn.GELU(), nn.Conv1d(width, width, 1), nn.Dropout(dropout)
        )
        self.restore = nn.ConvTranspose1d(width, width, scale, stride=scale, groups=width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        remainder = (-length) % self.scale
        temporal = hidden.transpose(1, 2)
        if remainder:
            temporal = torch.nn.functional.pad(temporal, (0, remainder), mode="replicate")
        downsampled = self.local(temporal)
        global_view = self.global_context(downsampled)
        restored = self.restore(downsampled + global_view)[..., :length].transpose(1, 2)
        return self.norm(hidden + restored)


class MICLayer(nn.Module):
    def __init__(self, width, scales, dropout):
        super().__init__()
        self.branches = nn.ModuleList([IsometricConvolutionBranch(width, scale, dropout) for scale in scales])
        self.merge = nn.Sequential(nn.Linear(width * len(scales), width), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden):
        return self.norm(hidden + self.merge(torch.cat([branch(hidden) for branch in self.branches], -1)))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, c_out=None, label_len=0,
                 features="M", d_model=64, d_layers=1, dropout=0.05,
                 embed="timeF", freq="h", conv_kernel=(12, 16)):
        super().__init__()
        c_out = enc_in if c_out is None else c_out
        if c_out != enc_in:
            raise ValueError("clean-room MICN requires c_out == enc_in")
        scales = tuple(int(kernel) for kernel in conv_kernel)
        if not scales or min(scales) < 2:
            raise ValueError("conv_kernel must contain scales >= 2")
        odd_kernels = tuple(kernel if kernel % 2 else kernel + 1 for kernel in scales)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.decomposition = MultiScaleDecomposition(odd_kernels)
        self.embedding = nn.Linear(enc_in, d_model)
        self.layers = nn.ModuleList([MICLayer(d_model, scales, dropout) for _ in range(d_layers)])
        self.seasonal_head = nn.Linear(d_model, enc_in)
        self.history_to_horizon = nn.Linear(seq_len, pred_len)
        self.trend_to_horizon = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        seasonal, trend = self.decomposition(x_enc)
        hidden = self.embedding(seasonal)
        for layer in self.layers:
            hidden = layer(hidden)
        refined = self.seasonal_head(hidden) + seasonal
        seasonal_forecast = self.history_to_horizon(refined.transpose(1, 2)).transpose(1, 2)
        trend_forecast = self.trend_to_horizon(trend.transpose(1, 2)).transpose(1, 2)
        return seasonal_forecast + trend_forecast
