---
name: "graph_spectral"
kind: "component"
module: "tsflab.models._components.graph_spectral"
summary: "Scaled normalized Laplacian (2L/lambda_max - I, degenerate-safe) and exact-order Chebyshev polynomial stacks for spectral graph convolutions."
category: "graph"
input: "scaled_laplacian(adj_mx [nodes, nodes]); chebyshev_polynomials(matrix [nodes, nodes], order); chebyshev_supports(adj_mx [nodes, nodes], order)"
output: "scaled_laplacian [nodes, nodes] float32 numpy; chebyshev_polynomials [order, nodes, nodes] numpy; chebyshev_supports [order, nodes, nodes] float32 torch"
origin: "Chebyshev spectral graph convolution (Defferrard et al., NeurIPS 2016) as used by STGCN-style forecasters; implementation is repository-written and its source is not recorded in history"
origin_models: ["astgcn", "stgcn", "gclstm", "dstagnn"]
tags: ["adjacency", "chebyshev", "degenerate", "graph", "laplacian", "spectral", "scaled-laplacian"]
---

# graph_spectral

## Purpose

Prepares supports for Chebyshev graph convolution. For adjacency `A`, degree
`D = rowsum(A)`, the normalized Laplacian is `L = I - D^{-1/2} A D^{-1/2}` (inverse
root of zero degree set to 0) and the scaled Laplacian is
`L~ = 2 L / lambda_max - I`, where `lambda_max` is the largest absolute eigenvalue of `L`
(if `lambda_max < 1e-12`, `L~ = L - I`). The Chebyshev stack is
`T_0 = I, T_1 = L~, T_k = 2 L~ T_{k-1} - T_{k-2}`.

## Origin and granularity

Introduced in commit `1822ab5d` ("rewrite(models): verify graph ssm and physical
families"), a large commit that rewrote graph models (including `astgcn` and `gclstm`)
away from vendored upstream code. The commit message gives no rationale for this
file; the "degenerate" keyword and the test name indicate it was written to handle
isolated nodes and identity graphs. Which upstream code it replaced is not recorded.
Current importers are `astgcn`, `stgcn`, `gclstm`, and `dstagnn` (all call
`chebyshev_supports`), and `graph_utils` reuses `scaled_laplacian` and
`chebyshev_polynomials`. The boundary is the numeric support
construction only. The Chebyshev convolution layer, the choice of order `K`, where
supports are registered as buffers, and the temporal blocks stay model-local. Note a
second, simpler `lambda_rescaled_laplacian(adj, lambda_max=2.0)` exists in `adj_norm`; the two
differ (this one computes the true `lambda_max`, symmetrizes, and rejects non-finite input).

## Interface

All three are module-level functions (no `__all__`); no parameters or state.

- `scaled_laplacian(adj_mx, *, undirected=True) -> np.ndarray`: `adj_mx` is any array
  convertible to float64 `[N, N]`. Raises `ValueError` for non-square input or
  non-finite values. With `undirected=True` the adjacency is symmetrized with
  `max(A, A^T)` and `eigvalsh` is used; with `False`, row degrees of the directed matrix
  and general `eigvals` are used (complex parts are dropped if not negligible).
  `N = 0` returns an empty array. Returns float32.
- `chebyshev_polynomials(matrix, order) -> np.ndarray`: exactly `order` matrices
  `[order, N, N]` in the dtype of `matrix`, starting with identity. Raises
  `ValueError` for `order < 1` or non-square input.
- `chebyshev_supports(adj_mx, order, *, undirected=True) -> torch.Tensor`: composition of the two
  above, float32 tensor `[order, N, N]` on CPU. `order=1` returns only the identity.
  Note `order` counts polynomials, so a "K-th order" convention (`T_0..T_K`) needs `K + 1`.

## Invariants and equivalence evidence

- `tests/test_repository_contracts.py`
  (`test_graph_spectral_supports_handle_degenerate_graphs`): identity graph gives a
  finite scaled Laplacian, `chebyshev_polynomials(.., 1)` has shape `(1, 3, 3)`,
  `chebyshev_supports(.., 3)` has shape `(3, 3, 3)` and is finite, `order=0` raises.
- no fixture: there is no `.pt` fixture; consumer-level behaviour is covered by the
  models' own contract tests rather than a frozen numerical reference.
- Contract and numerical regression: `tests/test_component_contracts_graph.py` with reference values in `tests/fixtures/components/graph_spectral.pt`.

## Variants and options

- `undirected=True` (default) symmetrizes; `undirected=False` keeps direction.
- `order` is exact (not `K + 1`); the identity is always the first support.
- Supports are dense; sparse graphs are densified.

## When to use and when not to use

Use for Chebyshev-polynomial spatial convolutions over a fixed adjacency, especially
when the graph can contain isolated nodes or be an identity. Do not use for
random-walk or diffusion supports (see `graph_utils`), for learned or dynamic
graphs (the eigendecomposition is host-side NumPy at construction time, `O(N^3)`),
or for large `N` where a dense `[order, N, N]` stack is too big.

## Related components

`graph_utils` (list-based support API, `scalap` mode wraps this), `diffusion_conv`
(applies supports as repeated powers), `adaptive_node_embedding_adjacency` (learned
adjacency to feed in).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `scaled_laplacian(adj_mx: np.ndarray, *, undirected: bool=True)`
  Return a dense scaled normalized Laplacian for any finite square graph.
- `chebyshev_polynomials(matrix: np.ndarray, order: int)`
  Return exactly ``order`` Chebyshev polynomials, beginning with identity.
- `chebyshev_supports(adj_mx: np.ndarray, order: int, *, undirected: bool=True)`
  Build exactly ``order`` dense Chebyshev supports for an adjacency matrix.

```python
from tsflab.models._components.graph_spectral import scaled_laplacian, chebyshev_polynomials, chebyshev_supports
```

## Retrieval terms

`adjacency`, `chebyshev`, `degenerate`, `graph`, `laplacian`, `spectral`

## Current model consumers (4)

`astgcn`, `dstagnn`, `gclstm`, `stgcn`
<!-- component-card:generated:end -->
