---
name: "graph_utils"
kind: "component"
module: "moderntsf.models._components.graph_utils"
summary: "Graph supports, Laplacians, and Chebyshev bases."
---

# graph_utils

## Purpose

Graph supports, Laplacians, and Chebyshev bases.

Canonical adjacency supports for spatiotemporal forecasting models.

Implementation: [`__init__.py`](__init__.py)

## Public API

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.graph_utils
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `graph_utils` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adjacency`, `chebyshev`, `graph`, `laplacian`, `support`.

## Current model consumers

- [`d2stgnn`](../../d2stgnn/README.md)
- [`dcrnn`](../../dcrnn/README.md)
- [`dfdgcn`](../../dfdgcn/README.md)
- [`gwnet`](../../gwnet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
