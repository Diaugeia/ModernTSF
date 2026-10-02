---
name: "regularized_adaptive_graph_conv"
kind: "component"
module: "tsflab.models._components.regularized_adaptive_graph_conv"
summary: "Row-swap regularizer for an embedding table plus a linear-in-nodes adaptive graph convolution that diffuses over an implicit cosine-style adjacency E'E'^T."
category: "graph"
input: "StochasticSharedEmbedding: embeddings [nodes, C]; EfficientCosineGraphConv: x [batch, nodes, hidden_dim]; node_embedding [nodes, spatial_dim]"
output: "StochasticSharedEmbedding: [nodes, C]; EfficientCosineGraphConv: [batch, nodes, hidden_dim]"
origin: "RAGC, 'Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution' (arXiv 2506.07179, 2025): stochastic shared embedding and Efficient Cosine Operator"
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

Added as new shared code with the `ragc` port; no earlier copies were consolidated. The
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

- `test_stochastic_shared_embedding` and `test_efficient_cosine_graph_conv` in
  `tests/test_component_contracts_graph.py` check: empty state dict for the
  regularizer, identity in eval or at `p=0`, every output row being an original row,
  gradient flow at `p=1`, `ValueError` for `p=1.5`; for the convolution, state-dict keys
  `gate_weight`, `filter_weight`, `out_weight`, `out_weight` shape `[H, (order + 1) H]`,
  support rows of unit L2 norm, output shape and dtype, the kernelized hop equal to the
  explicit `D^{-1} A x` (atol 1e-4), gradients to `x`, the embedding and all
  parameters, `ValueError` for a wrong `spatial_dim`, wrong `hidden_dim`, `order=0` and
  `hidden_dim=0`, and `dropout > 0` selecting `nn.Dropout`. Reference values:
  `tests/fixtures/components/regularized_adaptive_graph_conv.pt` and
  `tests/fixtures/components/stochastic_shared_embedding.pt`.
- `tests/test_local_graph_forecasters.py`
  (`test_ragc_efficient_cosine_operator_matches_dense_adjacency_matmul`) checks one
  kernelized hop equals the dense `(A x) / (A 1 + 1e-6)` computation at `atol=1e-5`;
  `test_ragc_stochastic_shared_embedding_regularizes_only_in_training` checks eval is
  identity and train output rows are always original rows.
- The fixtures were recorded from the extracted code; no official-code comparison is
  cited, so the match with the paper's operator is by structure only.

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

`adaptive_node_embedding_adjacency` (same "adjacency from node embeddings" responsibility, but it materializes a dense row-softmax `[N, N]` adjacency in `O(N^2)`; this one never forms it, uses a gated-cosine support, and also applies the diffusion),
`diffusion_conv` (dense static-support diffusion), `sparse_connection_router`
(sparse learned adjacency over positions, in contrast with this implicit dense
kernelized one), `graph_utils` (static supports that a model may combine with this learned
one), `node_visibility` (another way `visifold`-style models cut node-attention cost;
random subsampling, not a graph operator).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `StochasticSharedEmbedding(p: float=0.1)`
  Regularize an embedding table by randomly swapping whole rows.
- `EfficientCosineGraphConv(hidden_dim: int, spatial_dim: int, order: int=1, dropout: float=0.0)`
  Linear-complexity adaptive graph convolution over a node-embedding table.

```python
from tsflab.models._components.regularized_adaptive_graph_conv import StochasticSharedEmbedding, EfficientCosineGraphConv
```

## Retrieval terms

`adaptive`, `adjacency`, `cosine`, `embedding`, `graph`, `linear-complexity`, `node`, `regularization`, `stochastic`

## Current model consumers (1)

`ragc`
<!-- component-card:generated:end -->
