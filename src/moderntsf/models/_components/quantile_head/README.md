---
name: "quantile_head"
kind: "component"
module: "moderntsf.models._components.quantile_head"
summary: "Input-conditioned monotone quantile head with non-crossing outputs."
---

# quantile_head

## Purpose

Input-conditioned monotone quantile head with non-crossing outputs.

Shared monotone, non-crossing quantile output head.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `QuantileHead(quantile_levels: list[float], in_features: int=1)`
  Monotone quantile head producing a non-crossing (B, L, C, Q) grid.
- `validate_quantile_levels(values: list[float] | tuple[float, ...] | None)`
  Return a validated, strictly increasing quantile-level list.
- `DEFAULT_QUANTILE_LEVELS`
  Public module constant.

```python
from moderntsf.models._components.quantile_head import QuantileHead, validate_quantile_levels, DEFAULT_QUANTILE_LEVELS
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `quantile_head` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `monotone`, `non-crossing`, `probabilistic`, `quantile`.

## Current model consumers

- [`mqrnn`](../../mqrnn/README.md)
- [`quantile_dlinear`](../../quantile_dlinear/README.md)
- [`quantile_patchtst`](../../quantile_patchtst/README.md)
- [`tirex`](../../tirex/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
