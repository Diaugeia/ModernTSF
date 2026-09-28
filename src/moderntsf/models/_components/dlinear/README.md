---
name: "dlinear"
kind: "component"
module: "moderntsf.models._components.dlinear"
summary: "Moving-average decomposition and channel-wise linear forecasting backbone."
---

# dlinear

## Purpose

Moving-average decomposition and channel-wise linear forecasting backbone.

Series decomposition and linear forecasting backbone used by DLinear methods.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `DLinearBackbone(c_in: int, seq_len: int, pred_len: int, kernel_size: int=25, individual: bool=False)`
  DLinear sequence-to-sequence forecasting backbone.

```python
from moderntsf.models._components.dlinear import DLinearBackbone
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `dlinear` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `decomposition`, `linear`, `moving-average`, `seasonal`, `trend`.

## Current model consumers

- [`dlinear`](../../dlinear/README.md)
- [`latenttsf`](../../latenttsf/README.md)
- [`quantile_dlinear`](../../quantile_dlinear/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
