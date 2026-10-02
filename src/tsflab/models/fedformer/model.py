"""Clean-room Fourier FEDformer forecast implementation from the ICML paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.forecast_embedding import ForecastEmbedding
from tsflab.models._components.series_decomposition import SeriesDecomposition


def selected_modes(length: int, count: int, method: str) -> torch.Tensor:
    """Return deterministic low or paper-style random rFFT mode indices."""
    available = length // 2 + 1
    count = min(count, available)
    if method == "low":
        return torch.arange(count)
    if method != "random":
        raise ValueError("mode_select must be 'low' or 'random'")
    generator = torch.Generator().manual_seed(104729 + 37 * length + count)
    return torch.randperm(available, generator=generator)[:count].sort().values


class FrequencyEnhancedBlock(nn.Module):
    """Paper equations (3)-(4), using a head-local complex spectral kernel."""

    def __init__(self, d_model: int, n_heads: int, length: int, modes: int, mode_select: str) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        indices = selected_modes(length, modes, mode_select)
        self.register_buffer("mode_indices", indices, persistent=True)
        count = len(indices)
        scale = 1.0 / max(1, self.head_dim)
        self.input_projection = nn.Linear(d_model, d_model)
        self.weight_real = nn.Parameter(
            scale * torch.randn(n_heads, count, self.head_dim, self.head_dim)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(n_heads, count, self.head_dim, self.head_dim)
        )
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, width = values.shape
        projected = self.input_projection(values).view(
            batch, length, self.n_heads, self.head_dim
        )
        spectrum = torch.fft.rfft(projected, dim=1)
        indices = self.mode_indices[self.mode_indices < spectrum.shape[1]]
        selected = spectrum[:, indices]
        weight = torch.complex(
            self.weight_real[:, : len(indices)], self.weight_imag[:, : len(indices)]
        )
        transformed = torch.einsum("bmhi,hmio->bmho", selected, weight)
        output_spectrum = torch.zeros_like(spectrum)
        output_spectrum[:, indices] = transformed
        output = torch.fft.irfft(output_spectrum, n=length, dim=1)
        return self.output_projection(output.reshape(batch, length, width))


class FrequencyEnhancedAttention(nn.Module):
    """Paper equations (6)-(7): cross-attention in selected Fourier modes."""

    def __init__(self, d_model: int, n_heads: int, query_length: int, key_length: int, modes: int, mode_select: str) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.register_buffer(
            "query_modes", selected_modes(query_length, modes, mode_select), persistent=True
        )
        self.register_buffer(
            "key_modes", selected_modes(key_length, modes, mode_select), persistent=True
        )
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.last_attention: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, query_length, width = query.shape
        key_length = context.shape[1]
        q = self.query(query).view(batch, query_length, self.n_heads, self.head_dim)
        k = self.key(context).view(batch, key_length, self.n_heads, self.head_dim)
        v = self.value(context).view(batch, key_length, self.n_heads, self.head_dim)
        q_frequency = torch.fft.rfft(q, dim=1)
        k_frequency = torch.fft.rfft(k, dim=1)
        v_frequency = torch.fft.rfft(v, dim=1)
        q_indices = self.query_modes[self.query_modes < q_frequency.shape[1]]
        k_indices = self.key_modes[self.key_modes < k_frequency.shape[1]]
        selected_q = q_frequency[:, q_indices]
        selected_k = k_frequency[:, k_indices]
        selected_v = v_frequency[:, k_indices]
        scores = torch.einsum(
            "bmhd,bnhd->bhmn", selected_q, torch.conj(selected_k)
        ).real / math.sqrt(self.head_dim)
        weights = torch.tanh(scores)
        mixed = torch.einsum(
            "bhmn,bnhd->bmhd", weights.to(selected_v.dtype), selected_v
        )
        mixed = mixed / max(1, len(k_indices))
        output_spectrum = torch.zeros_like(q_frequency)
        output_spectrum[:, q_indices] = mixed
        self.last_attention = weights.detach()
        output = torch.fft.irfft(output_spectrum, n=query_length, dim=1)
        return self.output(output.reshape(batch, query_length, width))


def _feed_forward(d_model: int, d_ff: int, dropout: float, activation: str) -> nn.Sequential:
    nonlinear: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Linear(d_model, d_ff), nonlinear, nn.Dropout(dropout),
        nn.Linear(d_ff, d_model), nn.Dropout(dropout),
    )


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, length: int, modes: int, mode_select: str, d_ff: int, moving_avg: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.frequency_block = FrequencyEnhancedBlock(
            d_model, n_heads, length, modes, mode_select
        )
        self.decomposition_one = SeriesDecomposition(moving_avg)
        self.feed_forward = _feed_forward(d_model, d_ff, dropout, activation)
        self.decomposition_two = SeriesDecomposition(moving_avg)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        seasonal, _ = self.decomposition_one(values + self.frequency_block(values))
        seasonal, _ = self.decomposition_two(seasonal + self.feed_forward(seasonal))
        return seasonal


class FEDformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, query_length: int, key_length: int, modes: int, mode_select: str, d_ff: int, moving_avg: int, dropout: float, activation: str, c_out: int) -> None:
        super().__init__()
        self.self_frequency = FrequencyEnhancedBlock(
            d_model, n_heads, query_length, modes, mode_select
        )
        self.cross_frequency = FrequencyEnhancedAttention(
            d_model, n_heads, query_length, key_length, modes, mode_select
        )
        self.feed_forward = _feed_forward(d_model, d_ff, dropout, activation)
        self.decompositions = nn.ModuleList([SeriesDecomposition(moving_avg) for _ in range(3)])
        self.trend_projections = nn.ModuleList([nn.Linear(d_model, c_out, bias=False) for _ in range(3)])

    def forward(self, seasonal: torch.Tensor, memory: torch.Tensor, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seasonal, trend_one = self.decompositions[0](seasonal + self.self_frequency(seasonal))
        seasonal, trend_two = self.decompositions[1](seasonal + self.cross_frequency(seasonal, memory))
        seasonal, trend_three = self.decompositions[2](seasonal + self.feed_forward(seasonal))
        for projection, extracted in zip(self.trend_projections, (trend_one, trend_two, trend_three)):
            trend = trend + projection(extracted)
        return seasonal, trend


class Model(nn.Module):
    """Forecast-only Fourier FEDformer with progressive decomposition."""

    def __init__(self, seq_len: int, label_len: int, pred_len: int, enc_in: int, dec_in: int, c_out: int, d_model: int, n_heads: int, e_layers: int, d_layers: int, d_ff: int, moving_avg: int, dropout: float, activation: str = "gelu", mode_select: str = "random", modes: int = 32) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, dec_in, c_out) < 1:
            raise ValueError("lengths and channel counts must be positive")
        if label_len < 0 or label_len > seq_len:
            raise ValueError("label_len must be between zero and seq_len")
        if not (enc_in == dec_in == c_out):
            raise ValueError("clean-room FEDformer requires enc_in=dec_in=c_out")
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.channels = enc_in
        decoder_length = label_len + pred_len
        self.decomposition = SeriesDecomposition(moving_avg)
        self.encoder_embedding = ForecastEmbedding(enc_in, d_model, dropout)
        self.decoder_embedding = ForecastEmbedding(dec_in, d_model, dropout)
        self.encoder = nn.ModuleList([
            FEDformerEncoderLayer(d_model, n_heads, seq_len, modes, mode_select, d_ff, moving_avg, dropout, activation)
            for _ in range(e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([
            FEDformerDecoderLayer(d_model, n_heads, decoder_length, seq_len, modes, mode_select, d_ff, moving_avg, dropout, activation, c_out)
            for _ in range(d_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.seasonal_projection = nn.Linear(d_model, c_out)

    @staticmethod
    def _default_marks(values: torch.Tensor, length: int) -> torch.Tensor:
        return values.new_zeros(values.shape[0], length, 6)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.channels})")
        decoder_length = self.label_len + self.pred_len
        if x_dec is not None and (x_dec.ndim != 3 or x_dec.shape[0] != x_enc.shape[0] or x_dec.shape[1:] != (decoder_length, self.channels)):
            raise ValueError("x_dec has an incompatible decoder contract")
        x_mark_enc = self._default_marks(x_enc, self.seq_len) if x_mark_enc is None else x_mark_enc
        x_mark_dec = self._default_marks(x_enc, decoder_length) if x_mark_dec is None else x_mark_dec
        if x_mark_enc.shape != (x_enc.shape[0], self.seq_len, 6):
            raise ValueError("encoder marks must align with x_enc and contain six columns")
        if x_mark_dec.shape != (x_enc.shape[0], decoder_length, 6):
            raise ValueError("decoder marks must align with the decoder sequence")

        seasonal_history, trend_history = self.decomposition(x_enc)
        mean_future = x_enc.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        zero_future = torch.zeros_like(mean_future)
        if self.label_len:
            seasonal = torch.cat((seasonal_history[:, -self.label_len :], zero_future), dim=1)
            trend = torch.cat((trend_history[:, -self.label_len :], mean_future), dim=1)
        else:
            seasonal, trend = zero_future, mean_future

        memory = self.encoder_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        seasonal = self.decoder_embedding(seasonal, x_mark_dec)
        for layer in self.decoder:
            seasonal, trend = layer(seasonal, memory, trend)
        seasonal = self.seasonal_projection(self.decoder_norm(seasonal))
        return (seasonal + trend)[:, -self.pred_len :]
