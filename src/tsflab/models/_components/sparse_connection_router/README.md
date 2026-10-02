---
name: "sparse_connection_router"
kind: "component"
module: "tsflab.models._components.sparse_connection_router"
summary: "Input-independent learned binary connection matrix per head over discrete positions: pair-MLP scores, top-k density with straight-through gradient and training Gumbel noise."
category: "routing"
input: "none (forward takes no tensor); constructor num_positions N, dim, heads"
output: "(connections [heads, N, N], probabilities [heads, N, N])"
origin: "Shared sparse connection matrix of LSINet (AAAI 2025, 'A Lightweight Sparse Interaction Network for Time Series Forecasting'): Multihead Sparse Interaction Mechanism / Shared Interaction Learning"
origin_models: ["lsinet"]
tags: ["adjacency", "bernoulli", "gumbel-softmax", "interaction", "shared", "sparse", "top-k", "straight-through"]
---

# sparse_connection_router

## Purpose

Learns, per head, a sparse 0/1 `N x N` matrix saying which of `N` positions (e.g. time
patches) exchange information. The matrix depends only on learned parameters, never on
the input batch, so one matrix is shared across all samples and channels.

Per call: a learnable table (`N` positions x `memory_dim * heads`) is split per head and
passed through a 3-layer linear encoder; every ordered pair `(i, j)` is scored by an MLP
over `concat(m_i, m_j)` giving `scores [heads, N, N]`; `probabilities = sigmoid(scores)`.
The hard matrix keeps the top `k = max(1, round(density * N * N))` entries per head of
`scores` (plus `gumbel_scale * Gumbel` noise in training), returned with the
straight-through form `hard + (soft - soft.detach())` where `soft = sigmoid(gated_scores)`.

## Origin and granularity

Added in commit `dd63af6c` (automated intake of SDMixer, SEMixer, LSINet, Dualformer) as a
new shared component taken from the `lsinet` port; no earlier copies were consolidated. The
`lsinet` test file ties it to the paper's shared connection matrix (Eqs. 3-6). It is cut as
the router alone. Kept model-local in `lsinet`: patching (Eq. 9 stride geometry), applying the
matrix to values (`connections @ values`, per-head value projection), the time-invariant
and time-updating MLPs, integration blocks, and any use of `probabilities`
for regularization (the model currently discards them).

## Interface

`SharedSparseConnectionRouter(num_positions, dim, heads=1, density=0.15, memory_dim=None, gumbel_scale=1.0)`

- `num_positions` (int >= 1) and `heads` (int >= 1); `density` in `(0, 1]`; violations raise
  `ValueError`. `memory_dim` defaults to `dim` (`dim` is used only for this default).
  `gumbel_scale` (>= 0): noise multiplier in training; `0` disables noise.
- `forward() -> (connections, probabilities)`: no arguments, both `[heads, N, N]` float.
  `connections` is numerically 0/1 with value gradients flowing through the sigmoid;
  the number of ones per head is `round(density * N * N)` (at least 1), except that ties at
  the threshold would add extra ones. `probabilities` is the sigmoid of the noise-free scores.
- Parameters/state-dict keys: `memory.weight [N, memory_dim * heads]`,
  `memory_encoder.{0,1,2}.{weight,bias}` (Linear `memory_dim -> memory_dim//2 -> memory_dim//4 ->
  memory_dim`, minimum width 1, with no nonlinearities between them), and `pair_scorer.{0,2}.{weight,bias}`
  (Linear `2*memory_dim -> memory_dim`, ReLU, Linear `memory_dim -> 1`). The non-persistent buffer
  `_position_index` is not in the state dict.
- Memory cost is `O(heads * N^2 * memory_dim)` for the pair tensor. Randomness: the global torch RNG,
  only when `self.training` and `gumbel_scale > 0`; in eval `forward` is deterministic.

## Invariants and equivalence evidence

- `test_sparse_connection_router_contract` in `tests/test_component_contracts_graph.py`
  checks `[heads, N, N]` shapes and float32, that `_position_index` is not in the
  state dict, exactly `round(density * N^2)` ones per head in eval and in train mode
  (with Gumbel noise), values in `{0, 1}`, probabilities in `[0, 1]`, deterministic
  repeated eval calls, nonzero gradient to `memory.weight` and every `pair_scorer`
  parameter, the `ValueError` cases (`num_positions=0`, `heads=0`, `density` 0 and
  1.5), and `density=1.0` giving all ones; reference values are in
  `tests/fixtures/components/sparse_connection_router.pt`.
- `tests/test_lsinet_forecaster.py`:
  `test_sparse_connection_router_is_input_independent` (two eval calls are identical),
  `test_learned_interaction_matrix_is_binary_and_respects_target_density` (values in
  `{0, 1}`, per-head ones count `max(1, round(density * N^2))`, probabilities in `[0, 1]`),
  and `test_multihead_router_is_a_sparse_connection_router_component` (each block uses this class).
- no fixture: there is no pre-refactor copy to compare against (the component was
  created directly for `lsinet`), so the only fixture is the seeded self-reference above.
  Not tested: `memory_dim != dim`, `gumbel_scale=0`, and tie behavior at the threshold.

## Variants and options

- The `bernoulli` and `gumbel-softmax` tags are retrieval aliases: there is no
  Bernoulli sampling; selection is a top-k threshold, perturbed in training by
  additive standard Gumbel noise (not a Gumbel-softmax relaxation).
- `heads` for independent matrices; `density` for target sparsity; `memory_dim` for
  embedding width; `gumbel_scale=0` for deterministic training-time selection.
- The sample-independent design is the point: for content-dependent sparsity compute scores
  from `x` instead (not provided).

## When to use and when not to use

Use for a fixed number of positions where one learned sparse topology should be shared
across samples and channels, for example as a lightweight alternative to attention. Do not use
for variable `N`, for input-dependent routing, for large `N` (the pairwise tensor is
`N^2`), or when gradients must reach `x` (the router never sees it).

## Related components

- `graph_masked_attention`: dense attention biased by a given adjacency; this
  router instead learns the (sparse, binary) adjacency and does not apply it to values.
- `node_visibility`: random token subsampling and subgraph grouping, not learned.
- `adaptive_node_embedding_adjacency`: dense learned adjacency from node embeddings
  (`softmax(relu(E1 E2^T))`); the same "learned adjacency" responsibility, differing in
  being dense, per-node-embedding, and with no density target or discrete output.
- `regularized_adaptive_graph_conv`: also learns an input-independent adjacency from
  node embeddings, but applies it as a linear-complexity graph convolution.
- `soft_tree`, `topk_expert_router`: other differentiable discrete routing (over
  tree leaves / experts, input-conditioned).
- `topk_expert_attention`: selects keys per query inside attention; this router selects connections between a learned memory and tokens.
- `weight_set_router`: input-independent softmax mixture of weight sets, no sparsity.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `SharedSparseConnectionRouter(num_positions: int, dim: int, heads: int=1, density: float=0.15, memory_dim: int | None=None, gumbel_scale: float=1.0)`
  Learn a shared 0/1 connection matrix over ``num_positions`` positions.

```python
from tsflab.models._components.sparse_connection_router import SharedSparseConnectionRouter
```

## Retrieval terms

`adjacency`, `bernoulli`, `gumbel-softmax`, `interaction`, `shared`, `sparse`, `top-k`

## Current model consumers (1)

`lsinet`
<!-- component-card:generated:end -->
