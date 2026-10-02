---
name: "diffusion_conv"
kind: "component"
module: "tsflab.models._components.diffusion_conv"
summary: "Graph WaveNet diffusion convolution: concatenates x and its 1..order powers under each static support along channels, then a 1x1 conv projection and dropout."
category: "graph"
input: "x [batch, c_in, nodes, time]; supports list of support_len matrices [nodes, nodes]"
output: "[batch, c_out, nodes, time]"
origin: "Diffusion convolution of DCRNN (Li et al., ICLR 2018) as applied in Graph WaveNet (Wu et al., IJCAI 2019)"
origin_models: ["gwnet"]
tags: ["diffusion", "graph", "graph-wavenet", "support", "spatiotemporal", "multi-hop"]
---

# diffusion_conv

## Purpose

Mixes information across graph nodes at every time step. For supports
`S_1..S_m` (`m = support_len`) and hop count `K = order`, with `x` shaped
`[B, C, N, T]`:

```
out = Dropout( Conv1x1( concat_channels[ x, S_1 x, S_1^2 x, .., S_1^K x, S_2 x, .., S_m^K x ] ) )
```

where `S x` means `einsum("ncvl,vw->ncwl", x, S)` (the support is applied on the
node axis as `x @ S`, so row `v` of `S` is the source node). The concatenation has
`(K * m + 1) * c_in` channels.

## Origin and granularity

Extracted in commit `2315e4e1` ("refactor(components): extract diffusion convolution")
from the `gwnet` and `sttn` upstream wrappers (the commit touched both
`_upstream.py` files); today only `gwnet` still imports it. The cut covers the
neighbourhood einsum, the 1x1 projection and the hop concatenation. The commit
message is empty, so the reason for the boundary is inferred from the code. Kept model-local: building the supports (`graph_utils`,
`adaptive_node_embedding_adjacency`), the gated dilated temporal convolution, the
residual and skip wiring, and the choice of `support_len` and `order`.

## Interface

`DiffusionConv2d(c_in, c_out, dropout, support_len=3, order=2)`

- `c_in`, `c_out` (int >= 1): input and output channels.
- `dropout` (float in [0, 1)): applied with `F.dropout` to the projected output only
  while `self.training`; no range validation.
- `support_len` (int >= 1): exact number of supports `forward` must receive.
- `order` (int >= 1): hops per support. Raises `ValueError` otherwise.
- `forward(x, supports)`: `x` is `[B, c_in, N, T]`; `supports` is a list of
  `[N, N]` tensors; each is moved to `x.device` (not dtype). Raises `ValueError` when
  `len(supports) != support_len`. Output `[B, c_out, N, T]`.
- Parameters (state-dict keys): `mlp.mlp.weight` `[c_out, (order*support_len+1)*c_in, 1, 1]`
  and `mlp.mlp.bias` `[c_out]`. No buffers, no persistent state.

Helper modules also exported: `NeighborhoodConv2d()` with
`forward(x, support)` computing one `einsum("ncvl,vw->ncwl")` hop (no parameters), and
`PointwiseProjection(in_channels, out_channels)`, a `Conv2d` with kernel `(1, 1)`
stored as attribute `mlp` (hence the doubled `mlp.mlp` prefix).

## Invariants and equivalence evidence

- `tests/test_component_contracts_graph.py` checks the Interface shapes, dtype, errors, invariants, gradient flow, and seeded numerical regression against `tests/fixtures/components/diffusion_conv.pt`.
- `tests/test_repository_contracts.py`
  (`test_diffusion_conv_matches_explicit_support_expansion`) checks the output
  against the explicit expansion `[x, S1 x, S1^2 x, S2 x, S2^2 x]` followed by the
  projection, plus finite gradients; `test_extracted_component_catalog_surface`
  checks catalog registration.
- `tests/test_component_extraction_graph.py` compares the `gwnet` model (which
  uses `DiffusionConv2d` with `support_len=3, order=2`) against a frozen
  pre-extraction copy: identical state-dict keys, eval outputs and gradients.
- no fixture: there is no `.pt` fixture for this component.

## Variants and options

- `order` controls hop depth; hop terms are repeated powers of the same support, not
  Chebyshev polynomials.
- `support_len` can be 1 for a single graph. Adaptive supports can be mixed with
  static ones in the list. Powers of the same support are recomputed per call.
- Dropout is applied after the projection, not on the hop terms.

## When to use and when not to use

Use for `[B, C, N, T]` spatiotemporal features with a fixed list of dense `[N, N]`
supports (forward/reverse transition plus an adaptive one). Do not use for
`[B, N, C]` layouts without a time axis (reshape first), for sparse supports
(they are dense einsums, `O(N^2)` per hop), for dynamic per-sample supports (the
einsum `vw` is shared across the batch), or when the support count may vary at runtime.

## Related components

`graph_utils` (builds the supports), `adaptive_node_embedding_adjacency` (adaptive
support), `gated_dilated_conv` (the temporal half of a Graph WaveNet layer),
`graph_spectral` (Chebyshev supports), `regularized_adaptive_graph_conv` (linear-time alternative).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `DiffusionConv2d(c_in: int, c_out: int, dropout: float, support_len: int=3, order: int=2)`
  Concatenate zero-through-``order`` graph diffusion terms and project.

```python
from tsflab.models._components.diffusion_conv import DiffusionConv2d
```

## Retrieval terms

`diffusion`, `graph`, `graph-wavenet`, `support`, `spatiotemporal`

## Current model consumers (1)

`gwnet`
<!-- component-card:generated:end -->
