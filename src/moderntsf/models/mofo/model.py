"""Independent MoFo implementation from period-structured paper equations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from moderntsf.models._components.revin import RevIN


class RegulatedRelaxation(nn.Module):
    """Paper Eq. (9), a differentiable periodic distance regulator."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(1.0))
        self.beta_raw = nn.Parameter(torch.tensor(1.0))

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.alpha_raw)
        beta = F.softplus(self.beta_raw)
        return torch.sigmoid(-alpha * (distance - beta)) + (
            torch.exp(-distance) * torch.sigmoid(-alpha * beta)
        )


def period_structured_patches(values: torch.Tensor, period: int) -> torch.Tensor:
    """Discrete sampling: rows contain phase-aligned observations."""
    batch, length, channels = values.shape
    cycles = math.ceil(length / period)
    target = cycles * period
    if target > length:
        values = F.pad(values.transpose(1, 2), (target - length, 0),
                       mode="replicate").transpose(1, 2)
    return values.reshape(batch, cycles, period, channels).permute(0, 3, 2, 1)


class PeriodModulatedAttention(nn.Module):
    def __init__(self, width: int, heads: int, bias: bool = True) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by head")
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=bias)
        self.key = nn.Linear(width, width, bias=bias)
        self.value = nn.Linear(width, width, bias=bias)
        self.output = nn.Linear(width, width, bias=bias)
        self.modulator = RegulatedRelaxation()

    def forward(self, queries, memory, distance):
        batch, query_count, width = queries.shape
        memory_count = memory.shape[1]
        q = self.query(queries).reshape(batch, query_count, self.heads, self.head_width).transpose(1, 2)
        k = self.key(memory).reshape(batch, memory_count, self.heads, self.head_width).transpose(1, 2)
        v = self.value(memory).reshape(batch, memory_count, self.heads, self.head_width).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_width)
        regulation = self.modulator(distance).clamp_min(1e-8)
        if regulation.ndim == 2:
            regulation = regulation.view(1, query_count, memory_count)
        scores = scores + regulation.log().unsqueeze(1)
        attended = torch.matmul(scores.softmax(-1), v)
        attended = attended.transpose(1, 2).reshape(batch, query_count, width)
        return self.output(attended)


class MoFoLayer(nn.Module):
    def __init__(self, width: int, heads: int, bias: bool) -> None:
        super().__init__()
        self.attention = PeriodModulatedAttention(width, heads, bias)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feed_forward = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 2 * width, bias=bias), nn.GELU(),
            nn.Linear(2 * width, width, bias=bias),
        )

    def forward(self, queries, memory, distance):
        queries = self.norm_attention(queries + self.attention(queries, memory, distance))
        return self.norm_feed_forward(queries + self.feed_forward(queries))


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int,
                 d_model: int = 64, periodic: int = 24, head: int = 4,
                 d_layers: int = 1, bias: int = 1, cias: int = 1) -> None:
        super().__init__()
        if not cias:
            raise ValueError("local MoFo currently requires channel-independent sharing")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = periodic
        self.normalization = RevIN(enc_in, affine=False)
        self.embedding = nn.Linear(1, d_model, bias=bool(bias))
        self.future_queries = nn.Parameter(torch.empty(pred_len, d_model))
        nn.init.normal_(self.future_queries, std=d_model ** -0.5)
        self.layers = nn.ModuleList([
            MoFoLayer(d_model, head, bool(bias)) for _ in range(d_layers)
        ])
        self.projection = nn.Linear(d_model, 1, bias=bool(bias))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        normalized = self.normalization(x_enc, "norm")
        patches = period_structured_patches(normalized, self.period)
        batch, channels, phases, cycles = patches.shape
        memory = self.embedding(patches.unsqueeze(-1))
        future_index = torch.arange(self.pred_len, device=x_enc.device)
        future_phase = (self.seq_len + future_index) % self.period
        future_cycle = (self.seq_len + future_index) // self.period
        history_cycle = torch.arange(cycles, device=x_enc.device)
        distance = (future_cycle[:, None] - history_cycle[None, :]).abs().to(x_enc.dtype)

        selected = memory[:, :, future_phase, :, :]
        selected = selected.permute(0, 2, 1, 3, 4).reshape(batch * self.pred_len * channels,
                                                            cycles, -1)
        queries = self.future_queries.view(1, self.pred_len, 1, -1).expand(batch, -1, channels, -1)
        queries = queries.reshape(batch * self.pred_len * channels, 1, -1)
        expanded_distance = distance.view(1, self.pred_len, 1, cycles).expand(
            batch, -1, channels, -1
        ).reshape(batch * self.pred_len * channels, 1, cycles)
        for layer in self.layers:
            queries = layer(queries, selected, expanded_distance)
        prediction = self.projection(queries).reshape(batch, self.pred_len, channels)
        return self.normalization(prediction, "denorm")
