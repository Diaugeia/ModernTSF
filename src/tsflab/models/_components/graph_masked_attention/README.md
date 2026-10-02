---
name: "graph_masked_attention"
kind: "component"
module: "tsflab.models._components.graph_masked_attention"
summary: "Multi-head attention whose weights average a dense softmax with a softmax restricted to adjacency-permitted pairs; reduces to plain attention without a mask."
category: "attention"
input: "query [..., L_q, model_dim]; key, value [..., L_kv, model_dim]; adj_mask (optional bool) broadcastable to [..., num_heads, L_q, L_kv]"
output: "[..., L_q, model_dim]"
origin: "Global-local spatial attention (GLSAtt) of Extralonger (NeurIPS 2024 Workshop, 'Toward a Unified Perspective of Spatial-Temporal Factors for Extra-Long-Term Traffic Forecasting')"
origin_models: ["extralonger"]
tags: ["adjacency", "attention", "global", "graph", "local", "mask", "spatial", "multi-head"]
---

# graph_masked_attention

## Purpose

Scaled dot-product multi-head attention in which a fixed adjacency biases but does
not replace data-driven attention. With per-head
`scores = (Q K^T) / sqrt(head_dim)`:

```
global = softmax(scores)
local  = softmax(scores.masked_fill(~adj_mask, -inf))
attn   = (global + local) / 2          # adj_mask given
attn   = global                        # adj_mask is None
out    = out_proj(merge_heads(attn @ V))
```

## Origin and granularity

Added in commit `6663e2e0` (automated intake of VisiFold, Extralonger, ST-SSDL,
RAGC) as a new shared component for Extralonger's spatial route; the module docstring
cites the paper equation `GLSAtt = (Softmax(alpha_local) + Softmax(alpha_global)) V / 2`.
It is cut at the attention layer including the Q/K/V and output projections. Kept
model-local in `extralonger`: building `adj_mask` from the dense adjacency
(`(dense != 0) | eye`), the residual, layer norm and feed-forward wrapper, the
temporal and mixed routes (which call this layer with `adj_mask=None`), and all
embeddings. The generalization to other graph forecasters is a claim of the module
docstring, not evidenced by a second consumer.

## Interface

`GlobalLocalGraphAttention(model_dim, num_heads=8)`

- `model_dim` (int >= 1): total width; raises `ValueError` unless divisible by
  `num_heads`. `num_heads` (int >= 1): head count; `head_dim = model_dim // num_heads`.
- `forward(query, key, value, adj_mask=None)`: inputs `[..., L, model_dim]` with
  matching leading batch axes (for `extralonger`: `[B, T, N, D]` style layouts where
  the attended axis is second-to-last). Only the last two axes are attended. Heads are
  inserted before the length axis, so `adj_mask` must broadcast to
  `[..., num_heads, L_q, L_kv]` (a `[L_q, L_kv]` mask works). It is cast to bool on
  the score device; `True` marks an allowed pair.
- Returns `[..., L_q, model_dim]`.
- Parameters (state-dict keys): `fc_q`, `fc_k`, `fc_v`, `out_proj`, each `nn.Linear(model_dim, model_dim)`
  with `.weight`/`.bias`. No buffers; no dropout; stateless.
- Fully masked rows: a query row with no allowed key (an all-`False` mask row) gets
  an all-zero attention row (neither the dense nor the local term is applied), so its
  output is the `out_proj` bias; outputs and gradients stay finite. Rows with at
  least one allowed key are bit-identical to the plain formula. (`extralonger` ORs
  in the identity, so it never hits this case.)
- Quirk: the mask only changes the local half, so masked pairs still receive half of
  their dense probability; the layer is a soft bias, not a hard mask.

## Invariants and equivalence evidence

- `tests/test_local_graph_forecasters.py`
  (`test_extralonger_global_local_attention_matches_paper_equation`) recomputes the
  equation by hand from the layer's own projections and compares both the masked
  output and the mask-free (plain dense attention) output.
- `tests/test_component_numeric_fixes.py` checks that a fully masked row is finite
  (bias-only output, finite gradients) and that rows with visible keys are
  bit-identical to the hand-computed equation and unaffected by masking another row.
- no fixture: no `.pt` fixture or pre-refactor copy exists, because the component
  was created directly for `extralonger` rather than extracted from existing code.

## Variants and options

- `adj_mask=None`: plain multi-head attention (self- or cross-attention, since
  `L_q` and `L_kv` may differ).
- The 50/50 blend of the two softmaxes is fixed; there is no learnable mixing weight,
  temperature, or dropout.

## When to use and when not to use

Use to let a fixed graph bias attention over nodes while keeping long-range learned
affinity. Do not use for hard graph-restricted attention (use an additive `-inf` mask
with a standard attention layer), for large `L` (attention is dense, `O(L^2)`; see
`node_visibility` for subgraph grouping), or with graphs that can have isolated rows
unless self-loops are added.

## Related components

`self_attention_family` (plain attention layers), `node_visibility` (cuts the node
set before dense node attention), `adaptive_node_embedding_adjacency` and `graph_utils`
(sources of adjacency), `sparse_connection_router` (learned sparse mixing instead of attention).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `GlobalLocalGraphAttention(model_dim: int, num_heads: int=8)`
  Multi-head attention averaging global and adjacency-masked local scores.

```python
from tsflab.models._components.graph_masked_attention import GlobalLocalGraphAttention
```

## Retrieval terms

`adjacency`, `attention`, `global`, `graph`, `local`, `mask`, `spatial`

## Current model consumers (1)

`extralonger`
<!-- component-card:generated:end -->
