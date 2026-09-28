---
name: "adain_style_norm"
kind: "component"
module: "moderntsf.models._components.adain_style_norm"
summary: "Adaptive instance normalization rescaling features to externally supplied statistics."
---

# adain_style_norm

## Purpose

Adaptive instance normalization rescaling features to externally supplied statistics.

Adaptive instance normalization: rescale features to external statistics.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `AdaptiveInstanceNorm1d(eps: float=1e-05)`
  Normalize over the sequence axis, then rescale to given statistics.

```python
from moderntsf.models._components.adain_style_norm import AdaptiveInstanceNorm1d
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `adain_style_norm` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adain`, `adaptive`, `non-stationary`, `normalization`, `style`.

## Current model consumers

- [`canet`](../../canet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
