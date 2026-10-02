---
name: "graph_utils"
kind: "component"
module: "tsflab.models._components.graph_utils"
summary: "List-based adjacency support builders: normalize_adj_mx modes (normlap, scalap, symadj, transition, doubletransition, identity, origin), torch conversion, and Chebyshev polynomials."
category: "graph"
input: "adj_mx [nodes, nodes] array; adj_type string; matrix [nodes, nodes] with order int"
output: "list of [nodes, nodes] supports (numpy/COO/torch float32); cheb_poly returns [order, nodes, nodes]"
origin: "Support-list helpers common to DCRNN/Graph WaveNet-style traffic code (normalized adjacency, random-walk transition matrices); first appeared as a vendored external helper in the CauAir model port, source not recorded"
origin_models: ["gwnet", "dcrnn", "dfdgcn", "d2stgnn"]
tags: ["adjacency", "chebyshev", "graph", "laplacian", "support", "transition", "random-walk", "normalization", "support-list", "doubletransition"]
---

# graph_utils

## Purpose

Turns one adjacency matrix into the list of dense supports a graph forecaster
consumes. `normalize_adj_mx(adj, adj_type)` returns, per mode:

- `normlap`: `I - D^{-1/2} A D^{-1/2}`, degrees from the row sums of `A`, `A` not symmetrized, zero-degree rows give zeros in `D^{-1/2}`;
- `scalap`: scaled Laplacian `2L/lambda_max - I` from `graph_spectral.scaled_laplacian`, which first symmetrizes `A` as `max(A, A^T)` and uses the true largest `|eigenvalue|` of `L` (`L - I` if that is below 1e-12); unlike `normlap` it is therefore not the plain row-sum Laplacian of a directed `A`;
- `symadj`: `I - normlap = D^{-1/2} A D^{-1/2}` (same row-sum, unsymmetrized convention as `normlap`);
- `transition`: random walk `D^{-1} A` (zero-degree rows stay zero);
- `doubletransition`: `[D^{-1} A, D_r^{-1} A^T]` (forward and reverse walks, two supports);
- `identity`: `I`;
- `origin`: `A` with diagonal set to 1.

## Origin and granularity

The file began as an `_external/graph_utils.py` module in commit `3f6b3120` ("Port 16
non-duplicate CauAir models"), a vendored helper. It was then rebuilt as an independent
composition over `adj_norm` and `graph_spectral` (later history commits `3ac9e264`, `fba5fa99`, and
`17835348` "refactor(graph): reuse shared adjacency supports", which switched `d2stgnn`,
`dcrnn`, `dfdgcn`, `gwnet`, and `stgcn` to it; `stgcn` now uses `graph_spectral` directly). The module docstring says it is
"an independent composition of the repository's dense adjacency normalizers and
spectral helpers"; the original external source is not recorded in history. The boundary is
the list-of-supports interface (mode names follow the DCRNN/Graph WaveNet convention).
The actual matrix math lives in `adj_norm` and `graph_spectral`; what stays model-local is
how supports are used (stacking, adaptive supports, graph convolution layers).
Current consumers: `gwnet`, `dcrnn`, `dfdgcn`, `d2stgnn`, `st_ssdl`, `stdmae`.

## Interface

Module-level functions (the catalog lists `normalize_adj_mx`, `adj_to_supports`, `cheb_poly`; no state). All
inputs are cast to float64 numpy internally; outputs are float32.

- `normalize_adj_mx(adj_mx, adj_type, return_type="dense") -> list`: `adj_mx` must be a
  finite square array (`ValueError` otherwise). `adj_type` in the modes above;
  unknown raises `ValueError("unknown adjacency normalization: ...")`.
  `return_type="dense"` gives float32 numpy arrays, `"coo"` gives
  `scipy.sparse.coo_matrix` with the same values; other values raise `ValueError`.
  One support for every mode except `doubletransition` (two). The adjacency is not
  symmetrized for `transition` modes; `scalap` symmetrizes (via `graph_spectral`).
- `adj_to_supports(adj_mx, adj_type="doubletransition", device="cpu") -> list[torch.Tensor]`:
  dense float32 tensors on `device`.
- `cheb_poly(matrix, order) -> np.ndarray`: thin wrapper of
  `graph_spectral.chebyshev_polynomials`; exactly `order` polynomials starting from
  identity, `ValueError` for `order < 1`.

## Invariants and equivalence evidence

- `tests/test_component_contracts_graph.py` (`test_graph_utils_normalize_modes`,
  `test_graph_utils_values_and_errors`) checks, for all seven modes, the support count,
  `[N, N]` float32 finite shape, COO equal to dense, and `adj_to_supports` dtype;
  `symadj == I - normlap`, `origin` has unit diagonal, the reverse walk equals
  `transition_matrix(A.T)`, `cheb_poly` equals `chebyshev_polynomials`, the four
  `ValueError` cases (unknown mode, unknown `return_type`, non-square, non-finite), and a
  regression of `fwd`, `rev`, `scalap` against `tests/fixtures/components/graph_utils.pt`.
- `tests/test_repository_contracts.py`
  (`test_shared_adjacency_normalizers_are_finite`): `adj_to_supports` returns float32
  and equals `transition_matrix(A)` and `transition_matrix(A.T)` on a graph with an
  isolated node; `scalap` is finite on an identity graph; `cheb_poly(.., 0)` and an
  unknown `adj_type` raise.
- `tests/test_component_extraction_graph.py` runs `adj_to_supports` inside frozen
  reference models of `gwnet`, `dfdgcn`, `d2stgnn` and compares outputs and gradients
  against the extracted models.

## Variants and options

- `doubletransition` (default of `adj_to_supports`) for diffusion models (`gwnet`,
  `dcrnn`, `d2stgnn`, `dfdgcn`); `symadj` is used by `st_ssdl`; `scalap` for Chebyshev use.
- `return_type="coo"` for sparse consumers; dense is the default.
- For direct Chebyshev stacks as tensors use `graph_spectral.chebyshev_supports` instead.

## When to use and when not to use

Use when a model takes a precomputed adjacency and needs a standard list of
normalized supports. Do not use for learned/adaptive graphs (see
`adaptive_node_embedding_adjacency`), for GCN renormalization with self-loops
(`adj_norm.gcn_norm`, not exposed here), or when `lambda_max` must be user-specified
(`graph_spectral.scaled_laplacian` always computes the true value).

## Related components

`graph_spectral` (the Laplacian scaling and Chebyshev math this module wraps; use it
directly for tensor Chebyshev stacks, as `stgcn` does), `adj_norm` (the row-sum normalizers
behind the other modes), `diffusion_conv` (consumes `doubletransition` supports),
`adaptive_node_embedding_adjacency` (learned instead of precomputed adjacency).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `normalize_adj_mx(adj_mx: np.ndarray, adj_type: str, return_type: str='dense')`
  Return the requested finite adjacency supports.
- `adj_to_supports(adj_mx: np.ndarray, adj_type: str='doubletransition', device: str | torch.device='cpu')`
  Convert an adjacency matrix to dense float32 support tensors.
- `cheb_poly(matrix: np.ndarray, order: int)`
  Return exactly ``order`` Chebyshev polynomials, beginning with identity.

```python
from tsflab.models._components.graph_utils import normalize_adj_mx, adj_to_supports, cheb_poly
```

## Retrieval terms

`adjacency`, `chebyshev`, `graph`, `laplacian`, `support`

## Current model consumers (6)

`d2stgnn`, `dcrnn`, `dfdgcn`, `gwnet`, `st_ssdl`, `stdmae`
<!-- component-card:generated:end -->
