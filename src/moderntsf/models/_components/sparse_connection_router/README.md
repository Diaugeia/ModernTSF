---
name: "sparse_connection_router"
kind: "component"
module: "moderntsf.models._components.sparse_connection_router"
summary: "Shared, input-independent sparse connection routing over discrete positions."
---

# sparse_connection_router

## Purpose

Shared, input-independent sparse connection routing over discrete positions.

Shared, input-independent sparse connection routing over discrete positions.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `SharedSparseConnectionRouter(num_positions: int, dim: int, heads: int=1, density: float=0.15, memory_dim: int | None=None, gumbel_scale: float=1.0)`
  Learn a shared 0/1 connection matrix over ``num_positions`` positions.

```python
from moderntsf.models._components.sparse_connection_router import SharedSparseConnectionRouter
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `sparse_connection_router` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adjacency`, `bernoulli`, `gumbel-softmax`, `interaction`, `shared`, `sparse`, `top-k`.

## Current model consumers

- [`lsinet`](../../lsinet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
