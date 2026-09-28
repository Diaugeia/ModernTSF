---
name: "diffusion_conv"
kind: "component"
module: "moderntsf.models._components.diffusion_conv"
summary: "Graph-WaveNet diffusion concatenation and projection for static supports."
---

# diffusion_conv

## Purpose

Graph-WaveNet diffusion concatenation and projection for static supports.

Graph-WaveNet-style diffusion convolution over static support matrices.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `DiffusionConv2d(c_in: int, c_out: int, dropout: float, support_len: int=3, order: int=2)`
  Concatenate zero-through-``order`` graph diffusion terms and project.

```python
from moderntsf.models._components.diffusion_conv import DiffusionConv2d
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `diffusion_conv` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `diffusion`, `graph`, `graph-wavenet`, `support`, `spatiotemporal`.

## Current model consumers

- [`gwnet`](../../gwnet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
