---
name: "adj_norm"
kind: "component"
module: "moderntsf.models._components.adj_norm"
summary: "Dense adjacency normalization."
---

# adj_norm

## Purpose

Dense adjacency normalization.

Adjacency-matrix normalization utilities for graph forecasting models.

Implementation: [`__init__.py`](__init__.py)

## Public API

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.adj_norm
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `adj_norm` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adjacency`, `graph`, `laplacian`, `normalization`.

## Current model consumers

- No model currently declares this component directly.

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
