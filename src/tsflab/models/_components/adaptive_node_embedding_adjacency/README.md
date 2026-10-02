---
name: "adaptive_node_embedding_adjacency"
kind: "component"
module: "tsflab.models._components.adaptive_node_embedding_adjacency"
summary: "Dense row-stochastic adaptive adjacency from learnable node embeddings: softmax(relu(source @ target), dim=-1), with an implicit self-similarity form."
category: "graph"
input: "source [nodes, dim] or [..., nodes, dim]; target (optional) [dim, nodes] or [..., dim, nodes]"
output: "adjacency [nodes, nodes] (or [..., nodes, nodes]), each row sums to 1"
origin: "Self-adaptive adjacency of Graph WaveNet (Wu et al., IJCAI 2019) with the single-embedding self-similarity form of AGCRN (Bai et al., NeurIPS 2020); the exact origin of each call site is the consumer model, not recorded per-formula in history"
origin_models: ["gwnet", "dfdgcn", "d2stgnn", "agcrn", "himnet"]
tags: ["adaptive", "adjacency", "embedding", "graph", "node", "softmax", "relu", "row-stochastic"]
---

# adaptive_node_embedding_adjacency

## Purpose

Builds a learned dense graph over `N` nodes from node embeddings, with no
predefined graph needed:

- dual form: `A = softmax(relu(E1 @ E2), dim=-1)` with `E1: [N, d]`, `E2: [d, N]`;
- single form (`target is None`): `A = softmax(relu(E @ E^T), dim=-1)`, with the
  transpose taken over the last two axes so batched `[..., N, d]` embeddings work.

Each row of `A` is a distribution over the columns, so `A` can be used directly as a
random-walk style support (`A @ x`) or as a basis element in a polynomial expansion.

## Origin and granularity

Five spatiotemporal models carried a verbatim copy of this three-operation block;
commit `dd8d8196` ("refactor(components): gated_dilated_conv and
adaptive_node_embedding_adjacency") extracted it from `gwnet`, `dfdgcn`, `himnet`,
`d2stgnn` and `agcrn`. The dual-embedding form is the Graph WaveNet self-adaptive
adjacency; the single-embedding form is the node-adaptive graph used by AGCRN.
The commit message and the module docstring record the models but not a per-copy
paper citation, so the paper attributions above are the conventional ones and the
repository only verifies the model list. The cut stops at the adjacency: how the
embeddings are parameterised and initialised (e.g. `himnet` passes a batched
per-sample "meta" embedding), identity/Chebyshev bases stacked on the adjacency
(`agcrn`, `himnet`), and how the adjacency is mixed with static or dynamic
supports stay model-local. `mtgnn` has a different graph constructor and was
deliberately left out of the extraction.

## Interface

`adaptive_node_embedding_adjacency(source, target=None)` (pure function, no
parameters, buffers, or state):

- `source`: float tensor `[..., nodes, dim]`.
- `target`: optional float tensor used exactly as supplied with no implicit
  transpose, so it must already be `[..., dim, nodes]` (e.g. `gwnet` stores its
  target table as `[dim, nodes]`). Omit it for the self-similarity form.
- Returns `[..., nodes, nodes]` (or `[..., nodes, target_nodes]` if the target has a
  different node count) in the dtype/device of the inputs; rows sum to 1.
- It does no validation: shape mismatches surface as torch matmul errors.
- Differentiable w.r.t. both embeddings. Rows whose scores are all non-positive
  become uniform (`relu` gives zeros), not zero rows.

## Invariants and equivalence evidence

- `tests/test_component_contracts_graph.py` checks the Interface shapes, dtype, errors, invariants, gradient flow, and seeded numerical regression against `tests/fixtures/components/adaptive_node_embedding_adjacency.pt`.
- `tests/test_component_extraction_graph.py` holds frozen verbatim pre-extraction
  copies of `gwnet`, `dfdgcn`, `himnet`, `d2stgnn` and `agcrn` and asserts identical
  state-dict keys and shapes, identical eval outputs (`atol=1e-6`) and identical
  gradients for each (`test_gwnet_gated_dilated_conv_and_adaptive_adjacency_equivalence`,
  `test_dfdgcn_gated_dilated_conv_and_adaptive_adjacency_equivalence`,
  `test_himnet_adaptive_adjacency_equivalence`,
  `test_d2stgnn_adaptive_adjacency_equivalence`,
  `test_agcrn_adaptive_adjacency_equivalence`).
- no fixture: there is no `.pt` fixture and no dedicated unit test of the function
  itself; the equivalence tests above are the only coverage.
- Row-stochasticity follows from the final softmax and was confirmed on a small
  CPU tensor.

## Variants and options

- Dual form (`target` given): independent source/target tables, asymmetric
  adjacency (`gwnet`, `dfdgcn`, `d2stgnn`).
- Single form (`target=None`): symmetric scores before the softmax, supports
  batched embeddings (`agcrn`, `himnet`).
- There is no temperature, top-k sparsification, or normalization variant; those
  would be new components.

## When to use and when not to use

Use for a small-to-moderate node count where an `N x N` dense adjacency is
affordable and a self-learned, row-normalized graph is wanted alongside or instead
of a predefined one. Do not use for large `N` (memory is `O(N^2)`; see
`regularized_adaptive_graph_conv`), when a sparse graph is required (see
`sparse_connection_router`), or when a different graph constructor such as MTGNN's
top-k asymmetric one is intended.

## Related components

`diffusion_conv` (consumes such adjacencies as supports), `graph_utils` (static
supports mixed with this learned one), `regularized_adaptive_graph_conv` (linear-time
node-embedding graph), `gated_dilated_conv` (co-extracted in the same commit),
`sparse_connection_router` (learned sparse adjacency over positions).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `adaptive_node_embedding_adjacency(source: torch.Tensor, target: torch.Tensor | None=None)`
  Return ``softmax(relu(source @ target), dim=-1)``.

```python
from tsflab.models._components.adaptive_node_embedding_adjacency import adaptive_node_embedding_adjacency
```

## Retrieval terms

`adaptive`, `adjacency`, `embedding`, `graph`, `node`, `softmax`

## Current model consumers (6)

`adamshyper`, `agcrn`, `d2stgnn`, `dfdgcn`, `gwnet`, `himnet`
<!-- component-card:generated:end -->
