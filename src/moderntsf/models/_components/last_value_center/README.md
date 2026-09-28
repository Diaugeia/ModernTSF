---
name: "last_value_center"
kind: "component"
module: "moderntsf.models._components.last_value_center"
summary: "Detached last-observed-timestep centering and restoration for BLC histories."
---

# last_value_center

## Purpose

Detached last-observed-timestep centering and restoration for BLC histories.

NLinear-style last-observed-value centering for ``(B, L, C)`` histories.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `center_on_last_value(x: torch.Tensor)`
  Subtract the detached final timestep from every step of ``x``.
- `restore_last_value(head_output: torch.Tensor, level: torch.Tensor)`
  Add the centering ``level`` back onto a head's forecast.

```python
from moderntsf.models._components.last_value_center import center_on_last_value, restore_last_value
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `last_value_center` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `centering`, `detach`, `last-value`, `level`, `residual`.

## Current model consumers

- [`crossgnn`](../../crossgnn/README.md)
- [`nlinear`](../../nlinear/README.md)
- [`segrnn`](../../segrnn/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
