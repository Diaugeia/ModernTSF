"""FreDF: Frequency Dynamic Fusion forecaster (ACM MM 2024).

The forecast is cast as learning a per-frequency transfer function in the Fourier
domain.  Zero-padding the history by the horizon, every FDBlock takes the rFFT of
the embedded sequence over time, multiplies component ``m`` by its own complex
``D x D`` matrix ``H[m]`` (Y[k] = H * X[k], paper eq. 6), returns to the time
domain and sums the per-frequency outputs with a trainable weight vector ``W``
(eq. 14).  Because the inverse FFT is linear and ``W`` is a real scalar per
frequency, the sum of K single-component inverse transforms equals one inverse
transform of the weighted spectrum, which is how the blocks are evaluated here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.marks import adapt_tslib_marks, tslib_time_feature_dimension
from tsflab.models._components.revin import RevIN


class FrequencyDynamicFusionBlock(nn.Module):
    """One FDBlock over ``[batch, length, d_model]`` with ``K = length // 2 + 1`` bins."""

    def __init__(self, length: int, d_model: int) -> None:
        super().__init__()
        self.length = length
        self.bins = length // 2 + 1
        bound = 1.0 / math.sqrt(d_model)
        # Per-frequency complex transfer matrices stored as real (re, im) pairs.
        self.transfer = nn.Parameter(
            torch.empty(self.bins, d_model, d_model, 2).uniform_(-bound, bound)
        )
        # Frequency importance logits (eq. 14); the fusion weights are the softmax of
        # these, recomputed in every forward as in the official code.
        self.frequency_weight = nn.Parameter(torch.randn(self.bins))

    def fusion_weights(self) -> torch.Tensor:
        """Fusion weights on the simplex: ``softmax(frequency_weight)``."""
        return torch.softmax(self.frequency_weight, dim=0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values, dim=1, norm="ortho")  # [B, K, D]
        transfer = torch.view_as_complex(self.transfer)  # [K, D_out, D_in]
        transferred = torch.einsum("bkd,ked->bke", spectrum, transfer)
        fused = transferred * self.fusion_weights().to(transferred.real.dtype)[None, :, None]
        return torch.fft.irfft(fused.contiguous(), n=self.length, dim=1, norm="ortho")

    def decoupled_reference(self, values: torch.Tensor) -> torch.Tensor:
        """Literal Algorithm 1 form (K masked copies), kept to verify equivalence."""
        spectrum = torch.fft.rfft(values, dim=1, norm="ortho")
        transfer = torch.view_as_complex(self.transfer)
        total = torch.zeros_like(values)
        weights = self.fusion_weights()
        for m in range(self.bins):
            single = torch.zeros_like(spectrum)
            single[:, m] = spectrum[:, m] @ transfer[m].transpose(0, 1)
            total = total + torch.fft.irfft(single.contiguous(), n=self.length, dim=1, norm="ortho"
            ) * weights[m]
        return total


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        e_layers: int = 1,
        dropout: float = 0.1,
        freq: str = "h",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, e_layers) < 1:
            raise ValueError("FreDF dimensions must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.freq = freq
        length = seq_len + pred_len
        self.revin = RevIN(enc_in, affine=False)
        self.history_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.ReLU(), nn.Dropout(dropout)
        )
        self.embedding = DataEmbedding(
            enc_in,
            d_model,
            "timeF",
            freq,
            dropout,
            time_feature_dim=tslib_time_feature_dimension(freq),
        )
        self.dropout = nn.Dropout(dropout)
        # One block reused by every layer, as in the official code.
        self.e_layers = e_layers
        self.blocks = nn.ModuleList([FrequencyDynamicFusionBlock(length, d_model)])
        self.horizon_mlp = nn.Sequential(
            nn.Linear(pred_len, pred_len), nn.ReLU(), nn.Dropout(dropout)
        )
        self.projection = nn.Linear(d_model, enc_in)

    def _joint_marks(self, x_mark_enc, x_mark_dec):
        if x_mark_enc is None or x_mark_dec is None:
            return None
        if x_mark_dec.shape[1] < self.pred_len:
            raise ValueError("x_mark_dec must cover the forecast horizon")
        marks = torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len :]), dim=1)
        return adapt_tslib_marks(marks, embed_type="timeF", freq=self.freq)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        hidden = self.revin(x_enc, "norm")
        hidden = self.history_mlp(hidden.transpose(1, 2)).transpose(1, 2)
        padding = hidden.new_zeros(hidden.shape[0], self.pred_len, hidden.shape[2])
        hidden = torch.cat((hidden, padding), dim=1)  # zero-pad the unknown future
        hidden = self.embedding(hidden, self._joint_marks(x_mark_enc, x_mark_dec))
        hidden = self.dropout(hidden)
        block = self.blocks[0]
        for _ in range(self.e_layers):
            hidden = hidden + self.dropout(block(hidden))
        future = hidden[:, -self.pred_len :, :]
        future = self.horizon_mlp(future.transpose(1, 2)).transpose(1, 2)
        return self.revin(self.projection(future), "denorm")
