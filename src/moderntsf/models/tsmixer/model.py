"""Clean-room implementation of the basic TSMixer architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from moderntsf.models._components.channel_wise_linear import ChannelWiseLinear


class MixerBlock(nn.Module):
    """Paper time mixing followed by feature mixing, both residual."""

    def __init__(self, seq_len: int, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        normalized_shape = (seq_len, channels)
        self.time_norm = nn.LayerNorm(normalized_shape)
        self.feature_norm = nn.LayerNorm(normalized_shape)
        self.time_projection = nn.Linear(seq_len, seq_len)
        self.feature_in = nn.Linear(channels, hidden)
        self.feature_out = nn.Linear(hidden, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_input = self.time_norm(x).transpose(1, 2)
        time_delta = self.dropout(self.activation(self.time_projection(time_input)))
        x = x + time_delta.transpose(1, 2)
        feature_input = self.feature_norm(x)
        feature_delta = self.feature_out(
            self.dropout(self.activation(self.feature_in(feature_input)))
        )
        return x + self.dropout(feature_delta)


class Model(nn.Module):
    """Basic historical-target TSMixer from Appendix B.3.2."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, e_layers) <= 0:
            raise ValueError("all dimensions and layer count must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.blocks = nn.ModuleList(
            MixerBlock(seq_len, enc_in, d_model, dropout) for _ in range(e_layers)
        )
        self.projection = ChannelWiseLinear(seq_len, pred_len, enc_in)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("TSMixer expects (batch, configured seq_len, enc_in)")
        hidden = x_enc
        for block in self.blocks:
            hidden = block(hidden)
        return self.projection(hidden.transpose(1, 2)).transpose(1, 2)
