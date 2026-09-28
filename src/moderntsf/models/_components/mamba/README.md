---
name: "mamba"
kind: "component"
module: "moderntsf.models._components.mamba"
summary: "Kernel-free selective state-space mixer, normalization, and residual block."
---

# mamba

## Purpose

Kernel-free selective state-space mixer, normalization, and residual block.

Kernel-free selective state-space blocks shared by Mamba forecasters.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `RMSNorm(d_model: int, eps: float=1e-05)`
  Root-mean-square normalization over the final feature dimension.
- `MambaBlock(d_model: int, d_inner: int, dt_rank: int, d_conv: int, d_state: int)`
  Pure-PyTorch selective state-space mixer with a causal depthwise convolution.
- `MambaResidualBlock(d_model: int, d_inner: int, dt_rank: int, d_conv: int, d_state: int)`
  Pre-normalized residual wrapper around :class:`MambaBlock`.

```python
from moderntsf.models._components.mamba import RMSNorm, MambaBlock, MambaResidualBlock
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `mamba` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `mamba`, `mixer`, `rmsnorm`, `ssm`, `state-space`.

## Current model consumers

- [`bimamba`](../../bimamba/README.md)
- [`mambasimple`](../../mambasimple/README.md)
- [`s_mamba`](../../s_mamba/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
