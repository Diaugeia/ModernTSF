---
name: "regularized_adaptive_graph_conv"
kind: "component"
module: "moderntsf.models._components.regularized_adaptive_graph_conv"
summary: "Row-swap regularizer for an embedding table plus a linear-in-nodes adaptive graph convolution that diffuses over an implicit cosine-style adjacency E'E'^T."
category: "graph"
input: "StochasticSharedEmbedding: embeddings [nodes, C]; EfficientCosineGraphConv: x [batch, nodes, hidden_dim]; node_embedding [nodes, spatial_dim]"
output: "StochasticSharedEmbedding: [nodes, C]; EfficientCosineGraphConv: [batch, nodes, hidden_dim]"
origin: "RAGC, 'Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution' (arXiv 2025): stochastic shared embedding and Efficient Cosine Operator"
origin_models: ["ragc"]
tags: ["adaptive", "adjacency", "cosine", "embedding", "graph", "linear-complexity", "node", "regularization", "stochastic"]
---

# regularized_adaptive_graph_conv

## Purpose

Two pieces for a node-embedding adaptive-graph forecaster.

`StochasticSharedEmbedding(p)`: in training, each row of an `[N, C]` table is replaced
with probability `p` by a uniformly random row of the same table (full-row swap, not
element dropout); identity in eval or when `p == 0`.

`EfficientCosineGraphConv`: builds a support table
`E' = L2normalize_rows( softmax(relu(E Wg), dim=1) * relu(E Wf) )` (`[N, C]`) and
diffuses `x` over the implicit adjacency `A = E' E'^T` without forming it:
one hop is `D^{-1} E' (E'^T x)` with `D = E' (E'^T 1) + 1e-6`, in `O(N C H)`.
Output is `Linear(concat[x, hop_1(x), .., hop_order(x)])` (no bias) with optional dropout.

## Origin and granularity

Added in commit `6663e2e0` (automated intake of VisiFold, Extralonger, ST-SSDL, RAGC)
as new shared code from the `ragc` port; no earlier copies were consolidated. The
docstring names the paper's "Efficient Cosine Operator" and a "stochastic shared
embedding" regularizer. The cut is these two modules because both operate on a bare
node-embedding table and carry no spatiotemporal assumptions beyond a node axis.
Kept model-local in `ragc`: the embedding table itself, the per-window embedding,
feed-forward blocks between graph convolutions, the number of layers, `sse_p`/`use_sse`,
and the auxiliary losses.

## Interface

`StochasticSharedEmbedding(p=0.1)`: `p` in `[0, 1]` else `ValueError`. `forward(embeddings
[N, C])` returns the same shape; no parameters or buffers; uses the global torch RNG and
`self.training`. Rows may be replaced by themselves.

`EfficientCosineGraphConv(hidden_dim, spatial_dim, order=1, dropout=0.0)`:

- `hidden_dim` (`H`), `spatial_dim` (`C`) >= 1 and `order` >= 1 else `ValueError`.
  `dropout` > 0 adds `nn.Dropout`, otherwise `Identity`.
- Parameters: `gate_weight [C, C]` and `filter_weight [C, C]` (both initialised
  `randn`, so not scaled by fan-in), `out_weight [H, (order + 1) * H]` (kaiming
  uniform, `a = sqrt(5)`). No buffers.
- `compute_support(node_embedding [N, C]) -> [N, C]`: raises `ValueError` if the last axis
  is not `spatial_dim`.
- `forward(x [B, N, H], node_embedding [N, C]) -> [B, N, H]`: raises `ValueError` if
  `x.shape[-1] != hidden_dim`. Stateless apart from parameters; the support is recomputed on
  every call, so gradients reach the embedding table.

## Invariants and equivalence evidence

- `tests/test_local_graph_forecasters.py`
  (`test_ragc_efficient_cosine_operator_matches_dense_adjacency_matmul`) checks one
  kernelized hop equals the dense `(A x) / (A 1 + 1e-6)` computation at `atol=1e-5`;
  `test_ragc_stochastic_shared_embedding_regularizes_only_in_training` checks eval is
  identity and train output rows are always original rows.
- no fixture: no `.pt` fixture or pre-refactor copy exists; the component was created
  directly for `ragc`.

## Variants and options

- `order` sets the number of kernelized hops (all concatenated with the order-0 input).
- `dropout` default 0.0; the docstring notes the official reference uses 0.1.
- `StochasticSharedEmbedding` with `p=0` or in eval is a no-op; `ragc` can swap in
  `nn.Identity` when `use_sse` is false.
- Hops are non-negative weights over positive supports (the ReLU and softmax), so the
  implicit adjacency is symmetric and entrywise non-negative.

## When to use and when not to use

Use for large node counts where a dense `[N, N]` adaptive adjacency is too costly, and
when an embedding table benefits from row-swap regularization. Do not use if you need the
adjacency explicitly (it is never formed), directed/asymmetric propagation (`E'E'^T` is
symmetric), per-sample dynamic graphs (one support for the batch), or per-time-step
`[B, N, T, H]` layouts without reshaping.

## Related components

`adaptive_node_embedding_adjacency` (dense softmax adjacency from embeddings),
`diffusion_conv` (dense static-support diffusion), `sparse_connection_router`,
`graph_utils`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `StochasticSharedEmbedding(p: float=0.1)`
  Regularize an embedding table by randomly swapping whole rows.
- `EfficientCosineGraphConv(hidden_dim: int, spatial_dim: int, order: int=1, dropout: float=0.0)`
  Linear-complexity adaptive graph convolution over a node-embedding table.

```python
from moderntsf.models._components.regularized_adaptive_graph_conv import StochasticSharedEmbedding, EfficientCosineGraphConv
```

## Retrieval terms

`adaptive`, `adjacency`, `cosine`, `embedding`, `graph`, `linear-complexity`, `node`, `regularization`, `stochastic`

## Current model consumers (1)

`ragc`
<!-- component-card:generated:end -->
