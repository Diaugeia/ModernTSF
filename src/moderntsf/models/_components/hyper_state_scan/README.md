---
name: "hyper_state_scan"
kind: "component"
module: "moderntsf.models._components.hyper_state_scan"
summary: "Kernel-free scalar-state selective scan and a 2-D grid state mixer."
---

# hyper_state_scan

## Purpose

Kernel-free scalar-state selective scan and a 2-D grid state mixer.

Kernel-free scalar-state selective scan and a 2-D grid state mixer.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `diagonal_selective_scan(u: torch.Tensor, delta: torch.Tensor, a: torch.Tensor, b: torch.Tensor)`
  Evaluate a scalar-state selective scan over the trailing length axis.
- `GridStateMixer(channels: int, kernel_size: int=3)`
  Depthwise 2-D convolution mixing a channel-first state grid locally.

```python
from moderntsf.models._components.hyper_state_scan import diagonal_selective_scan, GridStateMixer
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `hyper_state_scan` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `grid`, `hyper-state`, `mamba`, `scan`, `ssm`, `state-space`.

## Current model consumers

- [`timepro`](../../timepro/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
