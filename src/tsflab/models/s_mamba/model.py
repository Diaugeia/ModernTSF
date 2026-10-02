"""Clean-room S-Mamba implementation from the paper's forecasting algorithm."""

from __future__ import annotations

import math

from torch import nn

from tsflab.models._components.mamba import MambaBlock


class InvertedTokenization(nn.Module):
    """Equation (3): one whole lookback window becomes each variate token."""

    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        return self.dropout(self.projection(values.transpose(1, 2)))


class SMambaLayer(nn.Module):
    """Bidirectional inter-variate Mamba followed by the temporal FFN."""

    def __init__(self, d_model, d_state, d_ff, d_conv, expand, dropout, activation):
        super().__init__()
        kwargs = {
            "d_model": d_model,
            "d_inner": d_model * expand,
            "dt_rank": math.ceil(d_model / 16),
            "d_conv": d_conv,
            "d_state": d_state,
        }
        self.forward_scan = MambaBlock(**kwargs)
        self.backward_scan = MambaBlock(**kwargs)
        self.scan_norm = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), act, nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        forward = self.forward_scan(tokens)
        backward = self.backward_scan(tokens.flip(1)).flip(1)
        tokens = self.scan_norm(tokens + self.dropout(forward + backward))
        return self.ffn_norm(tokens + self.dropout(self.temporal_ffn(tokens)))


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=128,
        d_state=16,
        d_ff=128,
        e_layers=2,
        d_conv=2,
        expand=1,
        dropout=0.1,
        activation="gelu",
        use_norm=True,
    ):
        super().__init__()
        if (
            min(
                seq_len,
                pred_len,
                enc_in,
                d_model,
                d_state,
                d_ff,
                e_layers,
                d_conv,
                expand,
            )
            < 1
        ):
            raise ValueError("all S-Mamba dimensions and counts must be positive")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.use_norm = use_norm
        self.tokenization = InvertedTokenization(seq_len, d_model, dropout)
        self.layers = nn.ModuleList(
            [
                SMambaLayer(d_model, d_state, d_ff, d_conv, expand, dropout, activation)
                for _ in range(e_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]"
            )
        if self.use_norm:
            mean = x_enc.mean(1, keepdim=True).detach()
            scale = x_enc.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
            values = (x_enc - mean) / scale
        else:
            values = x_enc
        tokens = self.tokenization(values)
        for layer in self.layers:
            tokens = layer(tokens)
        forecast = self.output_projection(tokens).transpose(1, 2)
        return forecast * scale + mean if self.use_norm else forecast
