---
name: "adaptive_node_embedding_adjacency"
kind: "component"
module: "moderntsf.models._components.adaptive_node_embedding_adjacency"
summary: "Learnable node-embedding adaptive adjacency: softmax(relu(E1 @ E2^T))."
---

# adaptive_node_embedding_adjacency

## Purpose

Learnable node-embedding adaptive adjacency: softmax(relu(E1 @ E2^T)).

Learnable node-embedding adaptive adjacency: ``softmax(relu(E1 @ E2^T))``.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `adaptive_node_embedding_adjacency(source: torch.Tensor, target: torch.Tensor | None=None)`
  Return ``softmax(relu(source @ target), dim=-1)``.

```python
from moderntsf.models._components.adaptive_node_embedding_adjacency import adaptive_node_embedding_adjacency
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `adaptive_node_embedding_adjacency` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adaptive`, `adjacency`, `embedding`, `graph`, `node`, `softmax`.

## Current model consumers

- [`agcrn`](../../agcrn/README.md)
- [`d2stgnn`](../../d2stgnn/README.md)
- [`dfdgcn`](../../dfdgcn/README.md)
- [`gwnet`](../../gwnet/README.md)
- [`himnet`](../../himnet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
