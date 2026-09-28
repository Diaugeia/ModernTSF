---
name: "mixer_block"
kind: "component"
module: "moderntsf.models._components.mixer_block"
summary: "Pre-normalized residual time mixing then residual feature mixing (TSMixer basic block)."
---

# mixer_block

## Purpose

Pre-normalized residual time mixing then residual feature mixing (TSMixer basic block).

Canonical TSMixer-style time/feature residual mixing block.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `MixerBlock(seq_len: int, channels: int, hidden: int, dropout: float)`
  Paper time mixing followed by feature mixing, both residual.

```python
from moderntsf.models._components.mixer_block import MixerBlock
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `mixer_block` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `feature`, `gelu`, `layernorm`, `mixer`, `residual`, `time`.

## Current model consumers

- [`tsmixer`](../../tsmixer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
