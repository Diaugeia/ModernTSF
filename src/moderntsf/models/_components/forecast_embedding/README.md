---
name: "forecast_embedding"
kind: "component"
module: "moderntsf.models._components.forecast_embedding"
summary: "Value projection plus normalized six-column raw-calendar embedding."
---

# forecast_embedding

## Purpose

Value projection plus normalized six-column raw-calendar embedding.

Forecast value and raw-calendar embedding shared by decomposition models.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `RawCalendarEmbedding(d_model: int)`
  Project six raw calendar columns after fixed-range normalization.
- `ForecastEmbedding(channels: int, d_model: int, dropout: float)`
  Add projected values and normalized raw-calendar covariates.

```python
from moderntsf.models._components.forecast_embedding import RawCalendarEmbedding, ForecastEmbedding
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `forecast_embedding` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `calendar`, `covariate`, `embedding`, `forecast`, `value`.

## Current model consumers

- [`autoformer`](../../autoformer/README.md)
- [`fedformer`](../../fedformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
