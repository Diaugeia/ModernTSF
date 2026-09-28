---
name: "haar_dwt1d"
kind: "component"
module: "moderntsf.models._components.haar_dwt1d"
summary: "Lossless single-level Haar discrete wavelet transform and its inverse."
---

# haar_dwt1d

## Purpose

Lossless single-level Haar discrete wavelet transform and its inverse.

Lossless single-level Haar discrete wavelet transform along the last axis.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `HaarDWT1D()`
  Forward single-level Haar DWT along the last axis.
- `HaarIDWT1D()`
  Inverse single-level Haar DWT along the last axis.

```python
from moderntsf.models._components.haar_dwt1d import HaarDWT1D, HaarIDWT1D
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `haar_dwt1d` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `dwt`, `haar`, `sub-series`, `wavelet`.

## Current model consumers

- [`swift`](../../swift/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
