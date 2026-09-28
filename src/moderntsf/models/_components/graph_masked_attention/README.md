---
name: "graph_masked_attention"
kind: "component"
module: "moderntsf.models._components.graph_masked_attention"
summary: "Multi-head attention blending global dense scores with local adjacency-masked scores."
---

# graph_masked_attention

## Purpose

Multi-head attention blending global dense scores with local adjacency-masked scores.

Multi-head attention blending a global dense score matrix with a local adjacency-masked score matrix.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `GlobalLocalGraphAttention(model_dim: int, num_heads: int=8)`
  Multi-head attention averaging global and adjacency-masked local scores.

```python
from moderntsf.models._components.graph_masked_attention import GlobalLocalGraphAttention
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `graph_masked_attention` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adjacency`, `attention`, `global`, `graph`, `local`, `mask`, `spatial`.

## Current model consumers

- [`extralonger`](../../extralonger/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
