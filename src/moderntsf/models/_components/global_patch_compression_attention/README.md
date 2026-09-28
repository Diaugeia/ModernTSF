---
name: "global_patch_compression_attention"
kind: "component"
module: "moderntsf.models._components.global_patch_compression_attention"
summary: "Two-stage cross-patch attention that compresses patches into summaries before broadcasting."
---

# global_patch_compression_attention

## Purpose

Two-stage cross-patch attention that compresses patches into summaries before broadcasting.

Two-stage cross-patch attention with global-patch compression.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `GlobalPatchCompressionAttention(d_model: int, n_heads: int, d_ff: int, dropout: float=0.0)`
  Compress each group's patches into one summary, then attend through it.

```python
from moderntsf.models._components.global_patch_compression_attention import GlobalPatchCompressionAttention
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `global_patch_compression_attention` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `attention`, `compression`, `cross-patch`, `global`, `patch`, `sensor`, `transformer`.

## Current model consumers

- [`sensorformer`](../../sensorformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
