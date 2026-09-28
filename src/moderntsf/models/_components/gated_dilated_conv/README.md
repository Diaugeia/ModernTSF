---
name: "gated_dilated_conv"
kind: "component"
module: "moderntsf.models._components.gated_dilated_conv"
summary: "Causal dilated padding plus the WaveNet gated activation unit."
---

# gated_dilated_conv

## Purpose

Causal dilated padding plus the WaveNet gated activation unit.

WaveNet-style causal dilated gated activation unit.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `causal_pad(x: torch.Tensor, dilation: int, kernel_size: int)`
  Left-pad the trailing (temporal) axis for a causal dilated convolution.
- `gated_dilated_conv(x: torch.Tensor, filter_conv: nn.Module, gate_conv: nn.Module)`
  Apply the WaveNet gated activation unit to ``x``.

```python
from moderntsf.models._components.gated_dilated_conv import causal_pad, gated_dilated_conv
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `gated_dilated_conv` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `causal`, `dilated`, `gate`, `gated-activation`, `wavenet`.

## Current model consumers

- [`dfdgcn`](../../dfdgcn/README.md)
- [`gwnet`](../../gwnet/README.md)
- [`mtgnn`](../../mtgnn/README.md)
- [`wavenet`](../../wavenet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
