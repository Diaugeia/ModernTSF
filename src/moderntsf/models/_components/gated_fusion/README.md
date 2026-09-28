---
name: "gated_fusion"
kind: "component"
module: "moderntsf.models._components.gated_fusion"
summary: "Learnable sigmoid gate that convexly blends two equal-shaped embeddings."
---

# gated_fusion

## Purpose

Learnable sigmoid gate that convexly blends two equal-shaped embeddings.

Learnable sigmoid gate that convexly blends two equal-shaped embeddings.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `GatedFusion(dim: int)`
  Blend two same-shaped embeddings with a learned per-position gate.

```python
from moderntsf.models._components.gated_fusion import GatedFusion
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `gated_fusion` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `fusion`, `gate`, `gated`, `mixture`, `sigmoid`.

## Current model consumers

- [`gateformer`](../../gateformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
