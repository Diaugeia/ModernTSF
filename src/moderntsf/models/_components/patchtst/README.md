---
name: "patchtst"
kind: "component"
module: "moderntsf.models._components.patchtst"
summary: "Patch extraction, time-series Transformer encoding, and PatchTST backbone."
---

# patchtst

## Purpose

Patch extraction, time-series Transformer encoding, and PatchTST backbone.

Independent channel-wise patch Transformer forecasting backbone.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `PatchTSTBackbone(c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, padding_patch: str | None, n_layers: int, d_model: int, n_heads: int, d_k: int | None, d_v: int | None, d_ff: int, activation: str, norm: str, attn_dropout: float, res_dropout: float, ffn_dropout: float, proj_dropout: float, head_dropout: float, pre_norm: bool, pe: str, learn_pe: bool, head_type: str, individual: bool, revin: bool, affine: bool, subtract_last: bool)`
  Patch each channel independently, encode patches, and forecast directly.

```python
from moderntsf.models._components.patchtst import PatchTSTBackbone
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `patchtst` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `backbone`, `channel-independent`, `patch`, `transformer`.

## Current model consumers

- [`quantile_patchtst`](../../quantile_patchtst/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
