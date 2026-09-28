---
name: "gaussian_parameter_head"
kind: "component"
module: "moderntsf.models._components.gaussian_parameter_head"
summary: "Independent Gaussian location/positive-scale parameter projection."
---

# gaussian_parameter_head

## Purpose

Independent Gaussian location/positive-scale parameter projection.

Gaussian location/scale projection with explicit positivity semantics.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `GaussianParameterHead(in_features: int, out_features: int, *, eps: float=1e-06, scale_transform: Literal['softplus', 'log1pexp']='softplus')`
  Project features to independent Gaussian location and positive scale.

```python
from moderntsf.models._components.gaussian_parameter_head import GaussianParameterHead
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `gaussian_parameter_head` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `distribution`, `gaussian`, `location`, `probabilistic`, `scale`.

## Current model consumers

- [`deepar`](../../deepar/README.md)
- [`gaussian_mlp`](../../gaussian_mlp/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
