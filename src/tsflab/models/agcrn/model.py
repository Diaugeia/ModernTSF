"""Local AGCRN implementation guided by the paper and official reference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.marks import to_spatiotemporal


def _adaptive_basis(nodes: torch.Tensor, order: int) -> torch.Tensor:
    adjacency = adaptive_node_embedding_adjacency(nodes)
    basis = [torch.eye(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)]
    if order > 1:
        basis.append(adjacency)
    for _ in range(2, order):
        basis.append(2 * adjacency @ basis[-1] - basis[-2])
    return torch.stack(basis)


class NodeAdaptiveConvolution(nn.Module):
    """Node-conditioned Chebyshev convolution from the AGCRN equation."""

    def __init__(self, in_dim: int, out_dim: int, order: int, node_dim: int) -> None:
        super().__init__()
        self.order = order
        self.weight_bank = nn.Parameter(torch.empty(node_dim, order, in_dim, out_dim))
        self.bias_bank = nn.Parameter(torch.empty(node_dim, out_dim))
        nn.init.xavier_uniform_(self.weight_bank)
        nn.init.zeros_(self.bias_bank)

    def forward(self, x: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        basis = _adaptive_basis(nodes, self.order)
        neighborhoods = torch.einsum("knm,bmc->bnkc", basis, x)
        weights = torch.einsum("nd,dkio->nkio", nodes, self.weight_bank)
        bias = nodes @ self.bias_bank
        return torch.einsum("bnki,nkio->bno", neighborhoods, weights) + bias


class AdaptiveGraphGRUCell(nn.Module):
    """GRU gates and candidate parameterized by adaptive graph filters."""

    def __init__(self, in_dim: int, hidden_dim: int, order: int, node_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        joint = in_dim + hidden_dim
        self.gates = NodeAdaptiveConvolution(joint, 2 * hidden_dim, order, node_dim)
        self.candidate = NodeAdaptiveConvolution(joint, hidden_dim, order, node_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat((x, state), dim=-1), nodes)
        ).chunk(2, dim=-1)
        proposal = torch.tanh(
            self.candidate(torch.cat((x, reset * state), dim=-1), nodes)
        )
        return update * state + (1.0 - update) * proposal


class Model(nn.Module):
    """Adaptive graph convolutional recurrent multi-step forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        rnn_units: int = 32,
        embed_dim: int = 8,
        num_layers: int = 1,
        cheb_k: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        del adj_mx
        if output_dim != 1:
            raise ValueError("TSFLab AGCRN exposes one value per node")
        if min(seq_len, pred_len, num_nodes, input_dim, rnn_units, embed_dim, num_layers, cheb_k) < 1:
            raise ValueError("AGCRN dimensions must be positive")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.cells = nn.ModuleList(
            AdaptiveGraphGRUCell(
                input_dim if layer == 0 else rnn_units,
                rnn_units,
                cheb_k,
                embed_dim,
            )
            for layer in range(num_layers)
        )
        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, rnn_units))
        self.readout = nn.Sequential(
            nn.Linear(2 * rnn_units, rnn_units),
            nn.GELU(),
            nn.Linear(rnn_units, 1),
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.horizon_embedding)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"AGCRN expects (B, {self.seq_len}, {self.num_nodes}) values")
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] < self.input_dim:
            raise ValueError("AGCRN received fewer input features than configured")
        states = [
            x_enc.new_zeros(x_enc.shape[0], self.num_nodes, cell.hidden_dim)
            for cell in self.cells
        ]
        for step in history[..., : self.input_dim].unbind(1):
            layer_input = step
            for index, cell in enumerate(self.cells):
                states[index] = cell(layer_input, states[index], self.node_embeddings)
                layer_input = states[index]
        final = states[-1].unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        horizon = self.horizon_embedding.view(1, self.pred_len, 1, -1).expand_as(final)
        return self.readout(torch.cat((final, horizon), dim=-1)).squeeze(-1)


__all__ = ["Model", "NodeAdaptiveConvolution", "AdaptiveGraphGRUCell"]
