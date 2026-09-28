---
name: "channel_wise_linear"
kind: "component"
module: "moderntsf.models._components.channel_wise_linear"
summary: "Shared or per-channel affine projection over the final sequence axis."
---

# channel_wise_linear

## Purpose

Shared or per-channel affine projection over the final sequence axis.

Linear projection over the last axis, shared or independent by channel.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `ChannelWiseLinear(input_length: int, output_length: int, channels: int, individual: bool=False)`
  Map ``(batch, channels, input_length)`` to a forecast length.

```python
from moderntsf.models._components.channel_wise_linear import ChannelWiseLinear
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `channel_wise_linear` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `channel-wise`, `forecast`, `individual`, `linear`, `projection`.

## Current model consumers

- [`cosa`](../../cosa/README.md)
- [`cyclenet`](../../cyclenet/README.md)
- [`distdf`](../../distdf/README.md)
- [`linear`](../../linear/README.md)
- [`mtsmixer`](../../mtsmixer/README.md)
- [`nlinear`](../../nlinear/README.md)
- [`rlinear`](../../rlinear/README.md)
- [`rpmixer`](../../rpmixer/README.md)
- [`tsmixer`](../../tsmixer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
