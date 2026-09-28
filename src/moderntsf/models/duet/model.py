"""Clean-room DUET: distributional temporal experts plus channel attention."""
from __future__ import annotations
import torch
from torch import nn
from moderntsf.models._components.revin import RevIN
from moderntsf.models._components.topk_expert_router import GatingMLP, topk_dense_mix


def moving_average(x, kernel):
    """Centered edge-padded moving average over time.

    Kept model-local rather than routed through
    ``moderntsf.models._components.series_decomposition.EdgePaddedMovingAverage``: that
    component validates ``kernel_size`` as strictly odd, and at least one
    other consumer (BiST) relies on that odd-only validation to reject
    misconfigured even kernel sizes. This helper additionally supports even
    kernels via asymmetric padding (``left=(kernel-1)//2``,
    ``right=kernel//2``), so extending the shared component's contract to
    match it would silently change that other consumer's error behavior.
    """
    if kernel <= 1:
        return x
    left, right = (kernel - 1) // 2, kernel // 2
    padded = torch.cat((x[:, :1].expand(-1, left, -1), x, x[:, -1:].expand(-1, right, -1)), 1)
    return torch.nn.functional.avg_pool1d(padded.transpose(1, 2), kernel, stride=1).transpose(1, 2)


class TemporalExpert(nn.Module):
    def __init__(self, seq_len, width, moving_avg, dropout):
        super().__init__()
        self.moving_avg = moving_avg
        self.trend = nn.Linear(seq_len, width)
        self.seasonal = nn.Linear(seq_len, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trend = moving_average(x, self.moving_avg)
        return self.dropout(self.trend(trend.transpose(1, 2)) + self.seasonal((x - trend).transpose(1, 2))).transpose(1, 2)


def mahalanobis_bias(series, epsilon=1e-4):
    """Negative pairwise Mahalanobis distance used as channel-attention bias."""
    centered = series - series.mean(-1, keepdim=True)
    variance = centered.square().mean(-1, keepdim=True) + epsilon
    scaled = centered / variance.sqrt()
    differences = scaled[:, :, None] - scaled[:, None, :]
    return -differences.square().mean(-1)


class ChannelAttention(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.heads, self.scale = heads, (width // heads) ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, tokens, bias):
        batch, channels, width = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, channels, 3, self.heads, width // self.heads).unbind(2)
        scores = torch.einsum("bchd,bkhd->bhck", q, k) * self.scale + bias[:, None]
        mixed = torch.einsum("bhck,bkhd->bchd", scores.softmax(-1), v).flatten(2)
        tokens = self.norm1(tokens + self.out(mixed))
        return self.norm2(tokens + self.ffn(tokens))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", d_model=512, n_heads=8,
                 e_layers=2, d_ff=2048, dropout=0.1, fc_dropout=0.1,
                 moving_avg=25, num_experts=4, k=2, hidden_size=256, noisy_gating=True):
        super().__init__()
        if d_model % n_heads or not 1 <= k <= num_experts:
            raise ValueError("invalid attention width or expert count")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in)
        self.k = k
        # Gate consumes the concatenated per-channel (mean, std) distributional
        # feature computed in ``forward``, hence ``2 * enc_in`` input features.
        self.router = GatingMLP(2 * enc_in, num_experts, hidden_size, noisy_gating)
        kernels = [max(2, moving_avg - 2 * i) for i in range(num_experts)]
        self.experts = nn.ModuleList([TemporalExpert(seq_len, d_model, kernel, fc_dropout) for kernel in kernels])
        self.channel_layers = nn.ModuleList([ChannelAttention(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.head = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        x = self.revin(x_enc, "norm")
        transposed = x.transpose(1, 2)
        features = torch.cat((transposed.mean(-1), transposed.std(-1, unbiased=False)), -1)
        weights = self.router(features)
        weights = topk_dense_mix(weights, self.k, 1e-3)
        expert_values = torch.stack([expert(x) for expert in self.experts], 1)
        tokens = torch.einsum("be,bedc->bcd", weights, expert_values)
        bias = mahalanobis_bias(x.transpose(1, 2))
        for layer in self.channel_layers:
            tokens = layer(tokens, bias)
        return self.revin(self.head(tokens).transpose(1, 2), "denorm")
