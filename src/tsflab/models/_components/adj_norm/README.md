---
name: "adj_norm"
kind: "component"
module: "tsflab.models._components.adj_norm"
summary: "Dense numpy adjacency normalizations: symmetric normalized and scaled Laplacian, GCN renormalization, and forward/reverse random-walk transition matrices."
category: "graph"
input: "adj [N, N] dense adjacency (array-like, any numeric dtype)"
output: "[N, N] float64 numpy array"
origin: "Standard graph-normalization definitions (Kipf and Welling GCN renormalization; DCRNN-style random-walk transitions), following BasicTS adjacent_matrix_norm.py (Apache-2.0)"
origin_models: []
tags: ["adjacency", "graph", "laplacian", "normalization", "gcn", "transition", "numpy", "stateless"]
---

# adj_norm

## Purpose

Pure-numpy helpers that turn a dense adjacency matrix `A` into the matrices that
spatiotemporal GNN layers consume. With `D` the diagonal of row sums:

- `symmetric_normalized_laplacian`: `L = I - D^{-1/2} A D^{-1/2}`.
- `lambda_rescaled_laplacian`: `(2 / lambda_max) * L - I` (Chebyshev rescaling).
- `gcn_norm`: `D~^{-1/2} (A + I) D~^{-1/2}` with degrees of `A + I`.
- `transition_matrix`: `D^{-1} A` (row-stochastic random walk).
- `reverse_transition_matrix`: `transition_matrix(A.T)`.

Zero-degree rows get inverse degree 0, so results stay finite (an isolated
node has Laplacian diagonal 1, transition row 0).

## Origin and granularity

The math is the published standard; the module docstring records that the
definitions are also used by BasicTS `adjacent_matrix_norm.py` (Apache-2.0). It
entered the repository with the "Batch A" commit (`b91231b9`, metrics, masked
losses, adjacency-norm utilities) as `models/_external/adj_norm.py`, and was
later moved under the shared components (`fba5fa99`, `33ea2050`). No model package
consumes it directly. Its consumer is the `graph_utils` component, which
composes it into named supports (`normlap`, `symadj`, `transition`,
`doubletransition`). Input validation of finite values, the choice among
normalization types, torch conversion, and the eigenvalue-based scaled
Laplacian stay in `graph_utils` / `graph_spectral`. No origin model is recorded.

## Interface

All functions take `adj` as any array-like convertible to a square 2-D array.
They return a new dense `[N, N]` `float64` numpy array and are stateless.
`ValueError` is raised for non-2-D or non-square input. NaN/inf in `adj` are not
checked. Non-symmetric inputs are allowed and not symmetrized.

- `symmetric_normalized_laplacian(adj)`: degrees from row sums.
- `lambda_rescaled_laplacian(adj, lambda_max=2.0)`: `lambda_max` (float, nonzero) is
  supplied by the caller; the default 2 gives `L - I`. It does not compute
  the eigenvalue and does not symmetrize.
- `gcn_norm(adj)`: adds self-loops (identity) before degrees, so existing
  diagonal entries are increased by 1.
- `transition_matrix(adj)`: rows sum to 1 except zero-degree rows (all 0).
- `reverse_transition_matrix(adj)`: normalizes by the in-degree
  (column sums of `adj`).
- Private helpers `_as_dense`, `_inv_sqrt_degree`, `_inv_degree` are not public.

## Invariants and equivalence evidence

- `test_shared_adjacency_normalizers_are_finite` in
  `tests/test_repository_contracts.py` checks `gcn_norm` and `transition_matrix`
  are finite on a graph with an isolated node, and that `adj_to_supports`
  returns `transition_matrix(A)` and `transition_matrix(A.T)` as float32.
- no fixture: no pre-refactor tensor fixture exists. The Laplacian, scaled
  Laplacian, and reverse-transition functions are only covered indirectly
  through `graph_utils` consumers (`dcrnn`, `gwnet`, `d2stgnn`, `dfdgcn`,
  `st_ssdl`) that run in the contract tests.

## Variants and options

- `lambda_rescaled_laplacian(adj, lambda_max)` (formerly `scaled_laplacian`, renamed to
  avoid clashing with `graph_spectral.scaled_laplacian`) is a different function: that one symmetrizes by default,
  rejects non-finite input, computes `lambda_max` from the spectrum, returns
  float32, and is what `graph_utils` uses for `scalap`. This one only rescales
  with the given `lambda_max`.

## When to use and when not to use

Use for small dense static graphs when a numpy-side normalization is needed
before converting to tensors. Do not use for large sparse graphs (it
materializes `N x N` dense matrices and diagonal matrices), for learned or
batched adjacencies (it is numpy, not differentiable), or when a validated,
float32, ready-made support list is wanted (use `graph_utils`).

## Related components

`graph_utils` (the consumer that builds supports), `graph_spectral`
(eigenvalue-scaled Laplacian and Chebyshev supports), `diffusion_conv`,
`adaptive_node_embedding_adjacency` (learned adjacency, already normalized by
softmax).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import tsflab.models._components.adj_norm
```

## Retrieval terms

`adjacency`, `graph`, `laplacian`, `normalization`

## Current model consumers (0)

No model currently declares this component directly.
<!-- component-card:generated:end -->
