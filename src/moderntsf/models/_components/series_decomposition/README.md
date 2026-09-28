---
name: "series_decomposition"
kind: "component"
module: "moderntsf.models._components.series_decomposition"
summary: "Edge-padded moving average and residual/trend decomposition for BLC data."
---

# series_decomposition

## Purpose

Edge-padded moving average and residual/trend decomposition for BLC data.

Edge-padded moving-average decomposition for ``(batch, time, channel)`` data.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `EdgePaddedMovingAverage(kernel_size: int, stride: int=1)`
  Smooth a series after repeating its first and last observations.
- `SeriesDecomposition(kernel_size: int)`
  Split ``(B, L, C)`` values into residual and moving-average trend.

```python
from moderntsf.models._components.series_decomposition import EdgePaddedMovingAverage, SeriesDecomposition
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `series_decomposition` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `decomposition`, `moving-average`, `residual`, `smoothing`, `trend`.

## Current model consumers

- [`amplifier`](../../amplifier/README.md)
- [`autoformer`](../../autoformer/README.md)
- [`bist`](../../bist/README.md)
- [`fedformer`](../../fedformer/README.md)
- [`micn`](../../micn/README.md)
- [`moderntcn`](../../moderntcn/README.md)
- [`stop`](../../stop/README.md)
- [`symtime`](../../symtime/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
