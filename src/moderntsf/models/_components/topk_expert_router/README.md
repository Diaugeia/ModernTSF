---
name: "topk_expert_router"
kind: "component"
module: "moderntsf.models._components.topk_expert_router"
summary: "Two-layer gating MLP with optional trainable noise and floor-blended top-k expert sparsification."
---

# topk_expert_router

## Purpose

Two-layer gating MLP with optional trainable noise and floor-blended top-k expert sparsification.

Gated top-k expert routing shared by DUET-style and DynamicTMoE-style mixtures.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `GatingMLP(in_features: int, experts: int, hidden: int, noisy: bool=True)`
  Two-layer gate: ``Linear -> GELU -> Linear`` with optional trainable noise.
- `topk_dense_mix(weights: torch.Tensor, k: int, floor: float)`
  Zero non-top-k experts, blend back a ``floor`` of the dense weights, renormalize.

```python
from moderntsf.models._components.topk_expert_router import GatingMLP, topk_dense_mix
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `topk_expert_router` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `expert`, `gate`, `gating`, `mixture`, `moe`, `routing`, `sparse`, `top-k`.

## Current model consumers

- [`duet`](../../duet/README.md)
- [`dynamic_tmoe`](../../dynamic_tmoe/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
