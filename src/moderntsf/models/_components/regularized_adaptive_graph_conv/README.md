---
name: "regularized_adaptive_graph_conv"
kind: "component"
module: "moderntsf.models._components.regularized_adaptive_graph_conv"
summary: "Stochastic embedding row-swap regularization plus a linear-complexity, node-embedding adaptive graph convolution (efficient cosine operator)."
---

# regularized_adaptive_graph_conv

## Purpose

Stochastic embedding row-swap regularization plus a linear-complexity, node-embedding adaptive graph convolution (efficient cosine operator).

Stochastic embedding regularization and a linear-complexity, node-embedding adaptive graph convolution ("efficient cosine operator").

Implementation: [`__init__.py`](__init__.py)

## Public API

- `StochasticSharedEmbedding(p: float=0.1)`
  Regularize an embedding table by randomly swapping whole rows.
- `EfficientCosineGraphConv(hidden_dim: int, spatial_dim: int, order: int=1, dropout: float=0.0)`
  Linear-complexity adaptive graph convolution over a node-embedding table.

```python
from moderntsf.models._components.regularized_adaptive_graph_conv import StochasticSharedEmbedding, EfficientCosineGraphConv
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `regularized_adaptive_graph_conv` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adaptive`, `adjacency`, `cosine`, `embedding`, `graph`, `linear-complexity`, `node`, `regularization`, `stochastic`.

## Current model consumers

- [`ragc`](../../ragc/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
