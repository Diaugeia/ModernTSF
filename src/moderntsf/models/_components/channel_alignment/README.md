---
name: "channel_alignment"
kind: "component"
module: "moderntsf.models._components.channel_alignment"
summary: "Slice or zero-pad the trailing feature axis to a requested width."
---

# channel_alignment

## Purpose

Slice or zero-pad the trailing feature axis to a requested width.

Deterministic trailing-channel alignment for model input adapters.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `fit_channels(values: torch.Tensor, width: int)`
  Slice or right-pad the final axis to exactly ``width`` channels.

```python
from moderntsf.models._components.channel_alignment import fit_channels
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `channel_alignment` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `adapter`, `channel`, `feature`, `padding`, `shape`.

## Current model consumers

- [`dcrnn`](../../dcrnn/README.md)
- [`gclstm`](../../gclstm/README.md)
- [`gts`](../../gts/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
