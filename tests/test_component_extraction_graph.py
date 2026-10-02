"""Behavior-preservation tests for the gated-dilated-conv / adaptive-graph
component extraction across the ``wavenet``, ``gwnet``, ``dfdgcn``,
``mtgnn``, ``himnet``, ``d2stgnn``, and ``agcrn`` models.

Each ``_Reference*`` class below is a frozen, verbatim copy of the model
exactly as it existed immediately before this extraction (duplicated
gate/pad/tanh/sigmoid and duplicated ``softmax(relu(E1 @ E2^T))`` blocks).
The tests build both the reference and the current (post-extraction) model
with the same seed and inputs, and assert:

* identical ``state_dict()`` keys and per-key shapes (so old checkpoints
  still load unchanged);
* identical outputs in eval mode (``torch.allclose``/``assert_close`` at
  ``atol=1e-6``);
* identical gradients for every parameter.

This freezes the pre-extraction behavior in the test itself, so equivalence
is checked in CI independent of git history.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.diffusion_conv import DiffusionConv2d
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.revin import RevIN

import tsflab.models.agcrn.model as agcrn_model
import tsflab.models.d2stgnn.model as d2stgnn_model
import tsflab.models.dfdgcn.model as dfdgcn_model
import tsflab.models.gwnet.model as gwnet_model
import tsflab.models.himnet.model as himnet_model
import tsflab.models.mtgnn.model as mtgnn_model
import tsflab.models.wavenet.model as wavenet_model


# ---------------------------------------------------------------------------
# Frozen reference copies (pre-extraction behavior)
# ---------------------------------------------------------------------------


class _RefWaveNetGatedCausalLayer(nn.Module):
    def __init__(self, residual_width, dilation_width, skip_width, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.filter = nn.Conv1d(residual_width, dilation_width, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(residual_width, dilation_width, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(dilation_width, residual_width, 1)
        self.skip = nn.Conv1d(dilation_width, skip_width, 1)

    def forward(self, values):
        padding = self.dilation * (self.kernel_size - 1)
        causal = F.pad(values, (padding, 0))
        gated = torch.tanh(self.filter(causal)) * torch.sigmoid(self.gate(causal))
        return values + self.residual(gated), self.skip(gated)


class _RefWaveNetModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        residual_channels=16,
        dilation_channels=16,
        skip_channels=64,
        end_channels=128,
        kernel_size=2,
        blocks=2,
        layers=2,
        use_norm=True,
    ):
        super().__init__()
        del label_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.revin = RevIN(enc_in, affine=True) if use_norm else None
        self.input_projection = nn.Conv1d(1, residual_channels, 1)
        self.causal_layers = nn.ModuleList(
            _RefWaveNetGatedCausalLayer(
                residual_channels, dilation_channels, skip_channels, kernel_size, 2**layer
            )
            for _ in range(blocks)
            for layer in range(layers)
        )
        self.final_skip = nn.Conv1d(residual_channels, skip_channels, 1)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(skip_channels, end_channels, 1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.horizon_projection = nn.Linear(end_channels, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        normalized = self.revin(x_enc, "norm") if self.revin is not None else x_enc
        batch = normalized.shape[0]
        values = normalized.transpose(1, 2).reshape(batch * self.enc_in, 1, self.seq_len)
        residual = self.input_projection(values)
        skips = []
        for layer in self.causal_layers:
            residual, skip = layer(residual)
            skips.append(skip)
        summary = self.head(torch.stack(skips).sum(0) + self.final_skip(residual)).squeeze(-1)
        forecast = self.horizon_projection(summary).view(batch, self.enc_in, self.pred_len).transpose(1, 2)
        if self.revin is not None:
            forecast = self.revin(forecast, "denorm")
        return forecast[..., -1:] if self.features == "MS" else forecast


class _RefWaveNetGraphLayer(nn.Module):
    def __init__(self, channels, skip, kernel, dilation, dropout):
        super().__init__()
        self.left_padding = dilation * (kernel - 1)
        self.filter = nn.Conv2d(channels, channels, (1, kernel), dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, kernel), dilation=(1, dilation))
        self.diffusion = DiffusionConv2d(channels, channels, dropout, support_len=3, order=2)
        self.residual = nn.Conv2d(channels, channels, 1)
        self.skip = nn.Conv2d(channels, skip, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x, supports):
        padded = F.pad(x, (self.left_padding, 0, 0, 0))
        gated = torch.tanh(self.filter(padded)) * torch.sigmoid(self.gate(padded))
        mixed = self.diffusion(gated, supports)
        hidden = self.norm(self.residual(mixed) + x)
        return hidden, self.skip(hidden)


class _RefGWNetModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        adj_mx=None,
        input_dim=3,
        dropout=0.3,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=64,
        end_channels=128,
        kernel_size=2,
        blocks=2,
        layers=2,
    ):
        super().__init__()
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        node_dim = min(10, max(2, num_nodes))
        self.source_nodes = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.target_nodes = nn.Parameter(torch.empty(node_dim, num_nodes))
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.input_projection = nn.Conv2d(input_dim, residual_channels, 1)
        self.layers = nn.ModuleList(
            _RefWaveNetGraphLayer(residual_channels, skip_channels, kernel_size, 2**layer, dropout)
            for _ in range(blocks)
            for layer in range(layers)
        )
        self.output = nn.Sequential(
            nn.ReLU(), nn.Conv2d(skip_channels, end_channels, 1), nn.ReLU(), nn.Conv2d(end_channels, pred_len, 1)
        )
        nn.init.xavier_uniform_(self.source_nodes)
        nn.init.xavier_uniform_(self.target_nodes)

    def graph_supports(self):
        adaptive = torch.softmax(torch.relu(self.source_nodes @ self.target_nodes), dim=-1)
        return [self.forward_support, self.reverse_support, adaptive]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        data = to_spatiotemporal(x_enc, x_mark_enc)
        hidden = self.input_projection(data[..., : self.input_dim].permute(0, 3, 2, 1))
        skip_total = None
        supports = self.graph_supports()
        for layer in self.layers:
            hidden, skip = layer(hidden, supports)
            skip_total = skip if skip_total is None else skip_total + skip
        return self.output(skip_total)[..., -1]


class _RefDynamicGraphMix(nn.Module):
    def __init__(self, channels, out_channels, order=2):
        super().__init__()
        self.order = order
        self.projection = nn.Conv2d(channels * (1 + 4 * order), out_channels, 1)

    @staticmethod
    def _apply(x, graph):
        if graph.ndim == 2:
            return torch.einsum("bcnt,nm->bcmt", x, graph)
        return torch.einsum("bcnt,bnm->bcmt", x, graph)

    def forward(self, x, graphs):
        terms = [x]
        for graph in graphs:
            value = self._apply(x, graph)
            terms.append(value)
            for _ in range(2, self.order + 1):
                value = self._apply(value, graph)
                terms.append(value)
        return self.projection(torch.cat(terms, dim=1))


class _RefFrequencyGraph(nn.Module):
    def __init__(self, seq_len, nodes, fft_dim, identity_dim, hidden):
        super().__init__()
        self.spectrum = nn.Linear(seq_len // 2 + 1, fft_dim)
        self.identity = nn.Parameter(torch.empty(nodes, identity_dim))
        self.query = nn.Linear(fft_dim + identity_dim, hidden)
        self.key = nn.Linear(fft_dim + identity_dim, hidden)
        nn.init.xavier_uniform_(self.identity)

    def forward(self, values):
        magnitude = torch.fft.rfft(values.transpose(1, 2), dim=-1).abs()
        identity = self.identity.unsqueeze(0).expand(values.shape[0], -1, -1)
        features = torch.cat((self.spectrum(magnitude), identity), dim=-1)
        scale = self.query.out_features**-0.5
        return torch.softmax(self.query(features) @ self.key(features).transpose(-1, -2) * scale, dim=-1)


class _RefDFDGCNModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        adj_mx=None,
        dropout=0.3,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=64,
        end_channels=128,
        kernel_size=2,
        blocks=2,
        layers=2,
        a=1.0,
        fft_emb=10,
        identity_emb=10,
        hidden_emb=30,
        subgraph=20,
    ):
        super().__init__()
        del a, subgraph
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        self.adaptive_source = nn.Parameter(torch.empty(num_nodes, hidden_emb))
        self.adaptive_target = nn.Parameter(torch.empty(hidden_emb, num_nodes))
        self.frequency_graph = _RefFrequencyGraph(seq_len, num_nodes, fft_emb, identity_emb, hidden_emb)
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_projection = nn.Conv2d(3, residual_channels, 1)
        count = blocks * layers
        self.filters = nn.ModuleList(
            nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, 2 ** (index % layers)))
            for index in range(count)
        )
        self.gates = nn.ModuleList(
            nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, 2 ** (index % layers)))
            for index in range(count)
        )
        self.graph_layers = nn.ModuleList(_RefDynamicGraphMix(residual_channels, residual_channels) for _ in range(count))
        self.skip_layers = nn.ModuleList(nn.Conv2d(residual_channels, skip_channels, 1) for _ in range(count))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.ReLU(), nn.Conv2d(skip_channels, end_channels, 1), nn.ReLU(), nn.Conv2d(end_channels, pred_len, 1)
        )
        nn.init.xavier_uniform_(self.adaptive_source)
        nn.init.xavier_uniform_(self.adaptive_target)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        data = to_spatiotemporal(x_enc, x_mark_enc)[..., :3]
        hidden = self.input_projection(data.permute(0, 3, 2, 1))
        adaptive = torch.softmax(torch.relu(self.adaptive_source @ self.adaptive_target), dim=-1)
        dynamic = self.frequency_graph(x_enc)
        graphs = [self.forward_support, self.reverse_support, adaptive, dynamic]
        skips = None
        for filter_layer, gate_layer, graph_layer, skip_layer in zip(
            self.filters, self.gates, self.graph_layers, self.skip_layers
        ):
            dilation = filter_layer.dilation[1]
            pad = dilation * (filter_layer.kernel_size[1] - 1)
            gated = torch.tanh(filter_layer(F.pad(hidden, (pad, 0, 0, 0)))) * torch.sigmoid(
                gate_layer(F.pad(hidden, (pad, 0, 0, 0)))
            )
            hidden = hidden + self.dropout(graph_layer(gated, graphs))
            skips = skip_layer(hidden) if skips is None else skips + skip_layer(hidden)
        return self.output(skips)[..., -1]


class _RefGraphConstructor(nn.Module):
    def __init__(self, nodes, dimension, top_k, alpha):
        super().__init__()
        self.left, self.right = nn.Parameter(torch.randn(nodes, dimension)), nn.Parameter(torch.randn(nodes, dimension))
        self.left_map, self.right_map = nn.Linear(dimension, dimension), nn.Linear(dimension, dimension)
        self.top_k, self.alpha = min(top_k, nodes), alpha

    def forward(self):
        left, right = torch.tanh(self.alpha * self.left_map(self.left)), torch.tanh(self.alpha * self.right_map(self.right))
        scores = torch.relu(torch.tanh(self.alpha * (left @ right.T - right @ left.T)))
        if self.top_k < scores.shape[0]:
            scores = scores * (scores >= scores.topk(self.top_k, -1).values[..., -1:])
        return scores / scores.sum(-1, keepdim=True).clamp_min(1e-6)


class _RefMixHop(nn.Module):
    def __init__(self, width, depth, alpha):
        super().__init__()
        self.depth, self.alpha = depth, alpha
        self.project = nn.Linear((depth + 1) * width, width)

    def forward(self, x, graph):
        states, current = [x], x
        for _ in range(self.depth):
            current = self.alpha * x + (1 - self.alpha) * torch.einsum("nm,btmd->btnd", graph, current)
            states.append(current)
        return self.project(torch.cat(states, -1))


class _RefMTGNNLayer(nn.Module):
    def __init__(self, width, graph_width, depth, alpha, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.filter, self.gate = (
            nn.Conv2d(width, graph_width, (1, 3), dilation=(1, dilation)),
            nn.Conv2d(width, graph_width, (1, 3), dilation=(1, dilation)),
        )
        self.forward_graph, self.backward_graph = _RefMixHop(graph_width, depth, alpha), _RefMixHop(graph_width, depth, alpha)
        self.residual, self.norm, self.dropout = nn.Linear(graph_width, width), nn.LayerNorm(width), nn.Dropout(dropout)

    def forward(self, x, graph):
        channels, pad = x.permute(0, 3, 2, 1), 2 * self.dilation
        padded = F.pad(channels, (pad, 0, 0, 0))
        temporal = (torch.tanh(self.filter(padded)) * torch.sigmoid(self.gate(padded))).permute(0, 3, 2, 1)
        spatial = self.forward_graph(temporal, graph) + self.backward_graph(temporal, graph.T)
        return self.norm(x + self.dropout(self.residual(spatial)))


class _RefMTGNNModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        adj_mx=None,
        input_dim=3,
        gcn_depth=2,
        subgraph_size=20,
        node_dim=40,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        layers=3,
        dropout=0.3,
        propalpha=0.05,
        tanhalpha=3.0,
        dilation_exponential=1,
        build_adj=True,
    ):
        super().__init__()
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim, self.build_adj = (
            seq_len,
            pred_len,
            num_nodes,
            input_dim,
            build_adj,
        )
        adj = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        adj = adj + np.eye(num_nodes, dtype=np.float32)
        adj /= np.maximum(adj.sum(-1, keepdims=True), 1e-6)
        self.register_buffer("predefined_graph", torch.from_numpy(adj))
        self.graph_constructor, self.graph_mix = (
            _RefGraphConstructor(num_nodes, node_dim, subgraph_size, tanhalpha),
            nn.Parameter(torch.tensor(0.5)),
        )
        self.input_projection = nn.Linear(input_dim, residual_channels)
        self.layers = nn.ModuleList(
            _RefMTGNNLayer(residual_channels, conv_channels, gcn_depth, propalpha, max(1, dilation_exponential**i), dropout)
            for i in range(layers)
        )
        self.skip = nn.Linear(residual_channels, skip_channels)
        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(skip_channels, end_channels), nn.GELU(), nn.Linear(end_channels, pred_len)
        )

    def learned_graph(self):
        if not self.build_adj:
            return self.predefined_graph
        weight = torch.sigmoid(self.graph_mix)
        graph = weight * self.graph_constructor() + (1 - weight) * self.predefined_graph
        return graph / graph.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        st = to_spatiotemporal(x_enc, x_mark_enc)
        if st.shape[-1] < self.input_dim:
            st = torch.cat((st, st.new_zeros(*st.shape[:-1], self.input_dim - st.shape[-1])), -1)
        state = self.input_projection(st[..., : self.input_dim])
        graph = self.learned_graph()
        for layer in self.layers:
            state = layer(state, graph)
        return self.head(self.skip(state[:, -1])).transpose(1, 2)


class _RefMetaGraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, order, meta_dim):
        super().__init__()
        self.order = order
        self.weight_bank = nn.Parameter(torch.empty(meta_dim, order, in_dim, out_dim))
        self.bias_bank = nn.Parameter(torch.empty(meta_dim, out_dim))
        nn.init.xavier_uniform_(self.weight_bank)
        nn.init.zeros_(self.bias_bank)

    def forward(self, x, meta):
        graph = torch.softmax(torch.relu(meta @ meta.transpose(-1, -2)), dim=-1)
        identity = torch.eye(meta.shape[1], device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)
        basis = [identity]
        if self.order > 1:
            basis.append(graph)
        for _ in range(2, self.order):
            basis.append(2 * graph @ basis[-1] - basis[-2])
        neighborhoods = torch.einsum("bknm,bmc->bnkc", torch.stack(basis, 1), x)
        weights = torch.einsum("bnd,dkio->bnkio", meta, self.weight_bank)
        bias = torch.einsum("bnd,do->bno", meta, self.bias_bank)
        return torch.einsum("bnki,bnkio->bno", neighborhoods, weights) + bias


class _RefMetaGraphGRUCell(nn.Module):
    def __init__(self, in_dim, hidden, order, meta_dim):
        super().__init__()
        self.hidden = hidden
        self.gates = _RefMetaGraphConvolution(in_dim + hidden, 2 * hidden, order, meta_dim)
        self.candidate = _RefMetaGraphConvolution(in_dim + hidden, hidden, order, meta_dim)

    def forward(self, x, state, meta):
        reset, update = torch.sigmoid(self.gates(torch.cat((x, state), -1), meta)).chunk(2, -1)
        proposal = torch.tanh(self.candidate(torch.cat((x, reset * state), -1), meta))
        return update * state + (1 - update) * proposal


class _RefHimNetModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        adj_mx=None,
        input_dim=3,
        output_dim=1,
        hidden_dim=32,
        num_layers=1,
        cheb_k=2,
        node_embedding_dim=8,
        st_embedding_dim=8,
        tod_embedding_dim=8,
        dow_embedding_dim=8,
        steps_per_day=288,
        use_teacher_forcing=True,
    ):
        super().__init__()
        del adj_mx, use_teacher_forcing
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.steps_per_day = steps_per_day
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, st_embedding_dim))
        context_dim = node_embedding_dim + tod_embedding_dim + dow_embedding_dim + st_embedding_dim
        self.meta_projection = nn.Linear(context_dim, node_embedding_dim)
        self.encoder = nn.ModuleList(
            _RefMetaGraphGRUCell(input_dim if layer == 0 else hidden_dim, hidden_dim, cheb_k, node_embedding_dim)
            for layer in range(num_layers)
        )
        self.decoder = nn.ModuleList(
            _RefMetaGraphGRUCell(1 if layer == 0 else hidden_dim, hidden_dim, cheb_k, node_embedding_dim)
            for layer in range(num_layers)
        )
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.horizon_embedding)

    def _meta(self, tod, dow, horizon):
        batch, nodes = tod.shape
        pieces = [
            self.node_embedding.unsqueeze(0).expand(batch, -1, -1),
            self.tod_embedding(tod),
            self.dow_embedding(dow),
            horizon.view(1, 1, -1).expand(batch, nodes, -1),
        ]
        return self.meta_projection(torch.cat(pieces, -1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec
        data = to_spatiotemporal(x_enc, x_mark_enc)
        states = [x_enc.new_zeros(x_enc.shape[0], self.num_nodes, cell.hidden) for cell in self.encoder]
        zero_horizon = self.horizon_embedding.new_zeros(self.horizon_embedding.shape[-1])
        for index, step in enumerate(data[..., : self.input_dim].unbind(1)):
            tod = (data[:, index, :, 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            dow = (data[:, index, :, 2] * 7).long().clamp(0, 6)
            meta = self._meta(tod, dow, zero_horizon)
            value = step
            for layer, cell in enumerate(self.encoder):
                states[layer] = cell(value, states[layer], meta)
                value = states[layer]
        decoder_input = x_enc[:, -1].unsqueeze(-1)
        outputs = []
        from tsflab.models._components.marks import normalized_time_features

        future = (
            normalized_time_features(x_mark_dec[:, -self.pred_len :])
            if x_mark_dec is not None and x_mark_dec.ndim == 3
            else None
        )
        for horizon in range(self.pred_len):
            if future is None:
                tod = torch.full(
                    (x_enc.shape[0], self.num_nodes), horizon % self.steps_per_day, device=x_enc.device, dtype=torch.long
                )
                dow = torch.zeros_like(tod)
            else:
                tod = (future[:, horizon, 0:1] * self.steps_per_day).long().expand(-1, self.num_nodes).clamp(
                    0, self.steps_per_day - 1
                )
                dow = (future[:, horizon, 1:2] * 7).long().expand(-1, self.num_nodes).clamp(0, 6)
            meta = self._meta(tod, dow, self.horizon_embedding[horizon])
            value = decoder_input
            for layer, cell in enumerate(self.decoder):
                states[layer] = cell(value, states[layer], meta)
                value = states[layer]
            decoder_input = self.output(value)
            outputs.append(decoder_input.squeeze(-1))
        return torch.stack(outputs, dim=1)


def _ref_propagate(x, graph):
    if graph.ndim == 2:
        return torch.einsum("blnc,nm->blmc", x, graph)
    return torch.einsum("blnc,bnm->blmc", x, graph)


class _RefDynamicGraphConstructor(nn.Module):
    def __init__(self, hidden, node_dim, nodes):
        super().__init__()
        self.node_source = nn.Parameter(torch.empty(nodes, node_dim))
        self.node_target = nn.Parameter(torch.empty(nodes, node_dim))
        self.query = nn.Linear(hidden + node_dim, node_dim)
        self.key = nn.Linear(hidden + node_dim, node_dim)
        nn.init.xavier_uniform_(self.node_source)
        nn.init.xavier_uniform_(self.node_target)

    def forward(self, hidden):
        summary = hidden.mean(dim=1)
        source = torch.cat((summary, self.node_source.unsqueeze(0).expand(summary.shape[0], -1, -1)), -1)
        target = torch.cat((summary, self.node_target.unsqueeze(0).expand(summary.shape[0], -1, -1)), -1)
        return torch.softmax(self.query(source) @ self.key(target).transpose(-1, -2) * self.query.out_features**-0.5, dim=-1)


class _RefDecoupledLayer(nn.Module):
    def __init__(self, hidden, graphs, spatial_order, temporal_kernel, dropout, *, with_backcast):
        super().__init__()
        self.spatial_order = spatial_order
        self.diffusion_projection = nn.Linear(hidden * (1 + graphs * spatial_order), hidden)
        self.inherent = nn.GRU(hidden, hidden, batch_first=True)
        self.temporal = nn.Conv1d(hidden, hidden, temporal_kernel, padding=temporal_kernel // 2)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.backcast = nn.Linear(2 * hidden, hidden) if with_backcast else None
        self.forecast = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual, graphs):
        terms = [residual]
        for graph in graphs:
            value = residual
            for _ in range(self.spatial_order):
                value = _ref_propagate(value, graph)
                terms.append(value)
        diffusion = torch.tanh(self.diffusion_projection(torch.cat(terms, -1)))
        batch, length, nodes, hidden = residual.shape
        inherent, _ = self.inherent(residual.transpose(1, 2).reshape(batch * nodes, length, hidden))
        inherent = (
            self.temporal(inherent.transpose(1, 2)).transpose(1, 2).reshape(batch, nodes, length, hidden).transpose(1, 2)
        )
        weight = torch.sigmoid(self.gate(torch.cat((diffusion, inherent), -1)))
        separated = torch.cat((weight * diffusion, (1 - weight) * inherent), -1)
        next_residual = residual
        if self.backcast is not None:
            next_residual = residual - self.dropout(self.backcast(separated))
        return next_residual, self.forecast(separated[:, -1])


class _RefD2STGNNModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        adj_mx=None,
        input_dim=3,
        num_feat=1,
        num_hidden=16,
        node_hidden=8,
        time_emb_dim=8,
        k_s=2,
        k_t=3,
        gap=1,
        num_layers=2,
        dropout=0.1,
        time_in_day_size=288,
        day_in_week_size=7,
        forecast_dim=64,
        output_hidden=128,
    ):
        super().__init__()
        del num_feat, output_hidden
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.time_in_day_size, self.day_in_week_size = time_in_day_size, day_in_week_size
        self.tod_embedding = nn.Embedding(time_in_day_size, time_emb_dim)
        self.dow_embedding = nn.Embedding(day_in_week_size, time_emb_dim)
        self.input_projection = nn.Linear(input_dim + 2 * time_emb_dim, num_hidden)
        self.graph = _RefDynamicGraphConstructor(num_hidden, node_hidden, num_nodes)
        self.adaptive_source = nn.Parameter(torch.empty(num_nodes, node_hidden))
        self.adaptive_target = nn.Parameter(torch.empty(node_hidden, num_nodes))
        self.layers = nn.ModuleList(
            _RefDecoupledLayer(num_hidden, 4, k_s, k_t, dropout, with_backcast=layer < num_layers - 1)
            for layer in range(num_layers)
        )
        self.forecast = nn.Sequential(
            nn.Linear(num_layers * num_hidden, forecast_dim), nn.ReLU(), nn.Linear(forecast_dim, pred_len)
        )
        nn.init.xavier_uniform_(self.adaptive_source)
        nn.init.xavier_uniform_(self.adaptive_target)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        data = to_spatiotemporal(x_enc, x_mark_enc)
        tod = (data[..., 1] * self.time_in_day_size).long().clamp(0, self.time_in_day_size - 1)
        dow = (data[..., 2] * self.day_in_week_size).long().clamp(0, self.day_in_week_size - 1)
        features = torch.cat((data[..., : self.input_dim], self.tod_embedding(tod), self.dow_embedding(dow)), -1)
        residual = self.input_projection(features)
        dynamic = self.graph(residual)
        adaptive = torch.softmax(torch.relu(self.adaptive_source @ self.adaptive_target), -1)
        graphs = [self.forward_support, self.reverse_support, adaptive, dynamic]
        forecasts = []
        for layer in self.layers:
            residual, partial = layer(residual, graphs)
            forecasts.append(partial)
        combined = torch.cat(forecasts, -1)
        return self.forecast(combined).transpose(1, 2)


def _ref_adaptive_basis(nodes, order):
    adjacency = torch.softmax(torch.relu(nodes @ nodes.transpose(0, 1)), dim=-1)
    basis = [torch.eye(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)]
    if order > 1:
        basis.append(adjacency)
    for _ in range(2, order):
        basis.append(2 * adjacency @ basis[-1] - basis[-2])
    return torch.stack(basis)


class _RefNodeAdaptiveConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, order, node_dim):
        super().__init__()
        self.order = order
        self.weight_bank = nn.Parameter(torch.empty(node_dim, order, in_dim, out_dim))
        self.bias_bank = nn.Parameter(torch.empty(node_dim, out_dim))
        nn.init.xavier_uniform_(self.weight_bank)
        nn.init.zeros_(self.bias_bank)

    def forward(self, x, nodes):
        basis = _ref_adaptive_basis(nodes, self.order)
        neighborhoods = torch.einsum("knm,bmc->bnkc", basis, x)
        weights = torch.einsum("nd,dkio->nkio", nodes, self.weight_bank)
        bias = nodes @ self.bias_bank
        return torch.einsum("bnki,nkio->bno", neighborhoods, weights) + bias


class _RefAdaptiveGraphGRUCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, order, node_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        joint = in_dim + hidden_dim
        self.gates = _RefNodeAdaptiveConvolution(joint, 2 * hidden_dim, order, node_dim)
        self.candidate = _RefNodeAdaptiveConvolution(joint, hidden_dim, order, node_dim)

    def forward(self, x, state, nodes):
        reset, update = torch.sigmoid(self.gates(torch.cat((x, state), dim=-1), nodes)).chunk(2, dim=-1)
        proposal = torch.tanh(self.candidate(torch.cat((x, reset * state), dim=-1), nodes))
        return update * state + (1.0 - update) * proposal


class _RefAGCRNModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        adj_mx=None,
        input_dim=3,
        rnn_units=32,
        embed_dim=8,
        num_layers=1,
        cheb_k=2,
        output_dim=1,
    ):
        super().__init__()
        del adj_mx
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.cells = nn.ModuleList(
            _RefAdaptiveGraphGRUCell(input_dim if layer == 0 else rnn_units, rnn_units, cheb_k, embed_dim)
            for layer in range(num_layers)
        )
        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, rnn_units))
        self.readout = nn.Sequential(nn.Linear(2 * rnn_units, rnn_units), nn.GELU(), nn.Linear(rnn_units, 1))
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.horizon_embedding)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        history = to_spatiotemporal(x_enc, x_mark_enc)
        states = [x_enc.new_zeros(x_enc.shape[0], self.num_nodes, cell.hidden_dim) for cell in self.cells]
        for step in history[..., : self.input_dim].unbind(1):
            layer_input = step
            for index, cell in enumerate(self.cells):
                states[index] = cell(layer_input, states[index], self.node_embeddings)
                layer_input = states[index]
        final = states[-1].unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        horizon = self.horizon_embedding.view(1, self.pred_len, 1, -1).expand_as(final)
        return self.readout(torch.cat((final, horizon), dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Equivalence harness
# ---------------------------------------------------------------------------


class ComponentExtractionEquivalenceTest(unittest.TestCase):
    def _assert_equivalent(self, ref_cls, new_cls, kwargs, inputs, seed=0):
        torch.manual_seed(seed)
        ref = ref_cls(**kwargs)
        torch.manual_seed(seed)
        new = new_cls(**kwargs)

        ref_state = ref.state_dict()
        new_state = new.state_dict()
        self.assertEqual(list(ref_state.keys()), list(new_state.keys()))
        for key in ref_state:
            self.assertEqual(ref_state[key].shape, new_state[key].shape)
            torch.testing.assert_close(ref_state[key], new_state[key], atol=1e-6, rtol=0)

        ref.eval()
        new.eval()
        ref_out = ref(*inputs)
        new_out = new(*inputs)
        self.assertEqual(ref_out.shape, new_out.shape)
        self.assertTrue(torch.isfinite(ref_out).all())
        torch.testing.assert_close(ref_out, new_out, atol=1e-6, rtol=0)

        ref_out.sum().backward()
        new_out.sum().backward()
        ref_grads = dict(ref.named_parameters())
        new_grads = dict(new.named_parameters())
        self.assertEqual(set(ref_grads), set(new_grads))
        for name, ref_param in ref_grads.items():
            new_param = new_grads[name]
            if ref_param.grad is None and new_param.grad is None:
                continue
            self.assertIsNotNone(ref_param.grad, msg=name)
            self.assertIsNotNone(new_param.grad, msg=name)
            torch.testing.assert_close(ref_param.grad, new_param.grad, atol=1e-6, rtol=0, msg=name)

    def test_wavenet_gated_dilated_conv_equivalence(self):
        x_enc = torch.randn(2, 24, 3)
        self._assert_equivalent(
            _RefWaveNetModel,
            wavenet_model.Model,
            dict(seq_len=24, pred_len=6, enc_in=3, blocks=2, layers=2),
            (x_enc, None, None, None),
        )

    def test_gwnet_gated_dilated_conv_and_adaptive_adjacency_equivalence(self):
        num_nodes = 4
        x_enc = torch.randn(2, 12, num_nodes)
        self._assert_equivalent(
            _RefGWNetModel,
            gwnet_model.Model,
            dict(seq_len=12, pred_len=12, num_nodes=num_nodes, blocks=1, layers=2),
            (x_enc, None, None, None),
        )

    def test_dfdgcn_gated_dilated_conv_and_adaptive_adjacency_equivalence(self):
        num_nodes = 4
        x_enc = torch.randn(2, 12, num_nodes)
        self._assert_equivalent(
            _RefDFDGCNModel,
            dfdgcn_model.Model,
            dict(seq_len=12, pred_len=12, num_nodes=num_nodes, blocks=1, layers=2),
            (x_enc, None, None, None),
        )

    def test_mtgnn_gated_dilated_conv_equivalence(self):
        num_nodes = 4
        x_enc = torch.randn(2, 12, num_nodes)
        self._assert_equivalent(
            _RefMTGNNModel,
            mtgnn_model.Model,
            dict(seq_len=12, pred_len=12, num_nodes=num_nodes, layers=2, gcn_depth=2),
            (x_enc, None, None, None),
        )

    def test_himnet_adaptive_adjacency_equivalence(self):
        num_nodes = 4
        x_enc = torch.randn(2, 12, num_nodes)
        self._assert_equivalent(
            _RefHimNetModel,
            himnet_model.Model,
            dict(seq_len=12, pred_len=6, num_nodes=num_nodes, hidden_dim=8, num_layers=1),
            (x_enc, None, None, None),
        )

    def test_d2stgnn_adaptive_adjacency_equivalence(self):
        num_nodes = 4
        x_enc = torch.randn(2, 12, num_nodes)
        self._assert_equivalent(
            _RefD2STGNNModel,
            d2stgnn_model.Model,
            dict(seq_len=12, pred_len=6, num_nodes=num_nodes, num_hidden=8, num_layers=2),
            (x_enc, None, None, None),
        )

    def test_agcrn_adaptive_adjacency_equivalence(self):
        num_nodes = 4
        x_enc = torch.randn(2, 12, num_nodes)
        self._assert_equivalent(
            _RefAGCRNModel,
            agcrn_model.Model,
            dict(seq_len=12, pred_len=6, num_nodes=num_nodes, rnn_units=8, num_layers=1),
            (x_enc, None, None, None),
        )


if __name__ == "__main__":
    unittest.main()
