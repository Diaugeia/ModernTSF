---
name: "topk_expert_attention"
kind: "component"
module: "moderntsf.models._components.topk_expert_attention"
summary: "Differentiable top-k local expert self-attention with an optional shared global expert."
---

# topk_expert_attention

## Purpose

Differentiable top-k local expert self-attention with an optional shared global expert.

Differentiable top-k local expert self-attention.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `TopKExpertAttention(dim: int, num_heads: int=8, topk: int=4, shared: bool=False, qk_dim: int | None=None, dropout: float=0.0)`
  Adaptive local self-attention over the top-``topk`` scoring positions.

```python
from moderntsf.models._components.topk_expert_attention import TopKExpertAttention
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `topk_expert_attention` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `attention`, `expert`, `mixture-of-experts`, `routing`, `top-k`.

## Current model consumers

- [`timeexpert`](../../timeexpert/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
