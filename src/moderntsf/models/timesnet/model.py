"""Clean-room TimesNet forecast implementation from the paper's 2D equations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moderntsf.models._components.dominant_periods import dominant_periods


class Inception2D(nn.Module):
    """Parameter-efficient average of odd square convolution kernels."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * index + 1,
                    padding=index,
                )
                for index in range(num_kernels)
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([kernel(values) for kernel in self.kernels], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    """Transform 1D variations into period-aligned 2D variations and aggregate."""

    def __init__(self, total_length: int, top_k: int, d_model: int, d_ff: int, num_kernels: int) -> None:
        super().__init__()
        self.total_length = total_length
        self.top_k = top_k
        self.convolution = nn.Sequential(
            Inception2D(d_model, d_ff, num_kernels),
            nn.GELU(),
            Inception2D(d_ff, d_model, num_kernels),
        )
        self.last_periods: torch.Tensor | None = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] != self.total_length:
            raise ValueError("TimesBlock received an unexpected temporal length")
        batch, length, channels = values.shape
        periods, amplitudes = dominant_periods(values, self.top_k)
        transformed = []
        for period_tensor in periods:
            period = int(period_tensor)
            padded_length = ((length + period - 1) // period) * period
            padded = F.pad(values, (0, 0, 0, padded_length - length))
            image = padded.reshape(batch, padded_length // period, period, channels)
            image = image.permute(0, 3, 1, 2).contiguous()
            encoded = self.convolution(image)
            transformed.append(
                encoded.permute(0, 2, 3, 1).reshape(batch, padded_length, channels)[:, :length]
            )
        stacked = torch.stack(transformed, dim=-1)
        weights = torch.softmax(amplitudes, dim=-1)[:, None, None, :]
        self.last_periods = torch.as_tensor(periods, device=values.device)
        return values + (stacked * weights).sum(dim=-1)


class CalendarEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(6, d_model, bias=False)

    def forward(self, marks: torch.Tensor) -> torch.Tensor:
        if marks.ndim != 3 or marks.shape[-1] != 6:
            raise ValueError("calendar marks must have shape (batch, time, 6)")
        scales = marks.new_tensor((2100.0, 12.0, 31.0, 6.0, 23.0, 59.0))
        return self.projection(marks / scales - 0.5)


class Model(nn.Module):
    """Forecast-only TimesNet with stacked residual TimesBlocks."""

    def __init__(self, seq_len: int, label_len: int, pred_len: int, enc_in: int, c_out: int, d_model: int, e_layers: int, d_ff: int, dropout: float, top_k: int = 3, num_kernels: int = 3) -> None:
        super().__init__()
        del label_len
        if min(seq_len, pred_len, enc_in, c_out, d_model, e_layers, d_ff) < 1:
            raise ValueError("lengths, channels, widths, and layer counts must be positive")
        if enc_in != c_out:
            raise ValueError("TimesNet normalization requires enc_in=c_out")
        total_length = seq_len + pred_len
        if top_k > total_length // 2:
            raise ValueError("top_k exceeds the available non-DC Fourier frequencies")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.value_embedding = nn.Linear(enc_in, d_model)
        self.calendar_embedding = CalendarEmbedding(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.temporal_projection = nn.Linear(seq_len, total_length)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(total_length, top_k, d_model, d_ff, num_kernels)
                for _ in range(e_layers)
            ]
        )
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.channels})")
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(x_enc.shape[0], self.seq_len, 6)
        if x_mark_enc.shape != (x_enc.shape[0], self.seq_len, 6):
            raise ValueError("encoder marks must align with x_enc and contain six columns")

        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev
        embedded = self.value_embedding(normalized) + self.calendar_embedding(x_mark_enc)
        embedded = self.embedding_dropout(embedded)
        encoded = self.temporal_projection(embedded.transpose(1, 2)).transpose(1, 2)
        for block, norm in zip(self.blocks, self.block_norms):
            encoded = norm(block(encoded))
        forecast = self.output_projection(encoded)
        forecast = forecast * stdev + mean
        return forecast[:, -self.pred_len :]
