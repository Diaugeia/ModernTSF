"""Local DFDGCN implementation from paper and official-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from moderntsf.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from moderntsf.models._components.gated_dilated_conv import gated_dilated_conv
from moderntsf.models._components.graph_utils import adj_to_supports
from moderntsf.models._components.marks import to_spatiotemporal


class DynamicGraphMix(nn.Module):
    """Mix static, adaptive, and per-sample frequency-domain neighborhoods."""

    def __init__(self, channels: int, out_channels: int, order: int = 2) -> None:
        super().__init__()
        self.order = order
        self.projection = nn.Conv2d(channels * (1 + 4 * order), out_channels, 1)

    @staticmethod
    def _apply(x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        if graph.ndim == 2:
            return torch.einsum("bcnt,nm->bcmt", x, graph)
        return torch.einsum("bcnt,bnm->bcmt", x, graph)

    def forward(self, x: torch.Tensor, graphs: list[torch.Tensor]) -> torch.Tensor:
        terms = [x]
        for graph in graphs:
            value = self._apply(x, graph)
            terms.append(value)
            for _ in range(2, self.order + 1):
                value = self._apply(value, graph)
                terms.append(value)
        return self.projection(torch.cat(terms, dim=1))


class FrequencyGraph(nn.Module):
    """Derive a directed graph from node spectra and identity embeddings."""

    def __init__(self, seq_len: int, nodes: int, fft_dim: int, identity_dim: int, hidden: int) -> None:
        super().__init__()
        self.spectrum = nn.Linear(seq_len // 2 + 1, fft_dim)
        self.identity = nn.Parameter(torch.empty(nodes, identity_dim))
        self.query = nn.Linear(fft_dim + identity_dim, hidden)
        self.key = nn.Linear(fft_dim + identity_dim, hidden)
        nn.init.xavier_uniform_(self.identity)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        magnitude = torch.fft.rfft(values.transpose(1, 2), dim=-1).abs()
        identity = self.identity.unsqueeze(0).expand(values.shape[0], -1, -1)
        features = torch.cat((self.spectrum(magnitude), identity), dim=-1)
        scale = self.query.out_features**-0.5
        return torch.softmax(self.query(features) @ self.key(features).transpose(-1, -2) * scale, dim=-1)


class Model(nn.Module):
    """Dilated temporal forecaster with frequency-derived dynamic graphs."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, dropout: float = 0.3, residual_channels: int = 16, dilation_channels: int = 16, skip_channels: int = 64, end_channels: int = 128, kernel_size: int = 2, blocks: int = 2, layers: int = 2, a: float = 1.0, fft_emb: int = 10, identity_emb: int = 10, hidden_emb: int = 30, subgraph: int = 20) -> None:
        super().__init__()
        del a, subgraph
        if residual_channels != dilation_channels:
            raise ValueError("local DFDGCN requires equal residual and dilation widths")
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adj_mx shape must match num_nodes")
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        self.adaptive_source = nn.Parameter(torch.empty(num_nodes, hidden_emb))
        self.adaptive_target = nn.Parameter(torch.empty(hidden_emb, num_nodes))
        self.frequency_graph = FrequencyGraph(seq_len, num_nodes, fft_emb, identity_emb, hidden_emb)
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_projection = nn.Conv2d(3, residual_channels, 1)
        count = blocks * layers
        self.filters = nn.ModuleList(nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, 2 ** (index % layers))) for index in range(count))
        self.gates = nn.ModuleList(nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, 2 ** (index % layers))) for index in range(count))
        self.graph_layers = nn.ModuleList(DynamicGraphMix(residual_channels, residual_channels) for _ in range(count))
        self.skip_layers = nn.ModuleList(nn.Conv2d(residual_channels, skip_channels, 1) for _ in range(count))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(skip_channels, end_channels, 1), nn.ReLU(), nn.Conv2d(end_channels, pred_len, 1))
        nn.init.xavier_uniform_(self.adaptive_source)
        nn.init.xavier_uniform_(self.adaptive_target)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"DFDGCN expects (B, {self.seq_len}, {self.num_nodes}) values")
        data = to_spatiotemporal(x_enc, x_mark_enc)[..., :3]
        hidden = self.input_projection(data.permute(0, 3, 2, 1))
        adaptive = adaptive_node_embedding_adjacency(self.adaptive_source, self.adaptive_target)
        dynamic = self.frequency_graph(x_enc)
        graphs = [self.forward_support, self.reverse_support, adaptive, dynamic]
        skips = None
        for filter_layer, gate_layer, graph_layer, skip_layer in zip(self.filters, self.gates, self.graph_layers, self.skip_layers):
            gated = gated_dilated_conv(hidden, filter_layer, gate_layer)
            hidden = hidden + self.dropout(graph_layer(gated, graphs))
            skips = skip_layer(hidden) if skips is None else skips + skip_layer(hidden)
        assert skips is not None
        return self.output(skips)[..., -1]


__all__ = ["Model", "DynamicGraphMix", "FrequencyGraph"]
