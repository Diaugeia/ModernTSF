---
name: "flatten_forecast_head"
kind: "component"
module: "moderntsf.models._components.flatten_forecast_head"
summary: "Shared or channel-wise linear forecast head over two flattened feature axes."
---

# flatten_forecast_head

## Purpose

Shared or channel-wise linear forecast head over two flattened feature axes.

Linear forecasting head for flattened feature and patch axes.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `FlattenForecastHead(individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float=0.0)`
  Map ``(B, C, D, P)``-like inputs to ``(B, C, horizon)``.

```python
from moderntsf.models._components.flatten_forecast_head import FlattenForecastHead
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `flatten_forecast_head` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `channel-wise`, `flatten`, `forecast`, `head`, `linear`, `patch`.

## Current model consumers

- [`gateformer`](../../gateformer/README.md)
- [`lsinet`](../../lsinet/README.md)
- [`semixer`](../../semixer/README.md)
- [`sensorformer`](../../sensorformer/README.md)
- [`srsnet`](../../srsnet/README.md)
- [`timeexpert`](../../timeexpert/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
