---
name: "differential_attention"
kind: "component"
module: "moderntsf.models._components.differential_attention"
summary: "Differential self-attention: the RMS-renormalized difference of two softmax attention maps."
---

# differential_attention

## Purpose

Differential self-attention: the RMS-renormalized difference of two softmax attention maps.

Differential self-attention.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `DifferentialAttention(d_model: int, num_heads: int, lambda_init: float=0.8, attention_dropout: float=0.0)`
  Full (non-causal) differential self-attention.

```python
from moderntsf.models._components.differential_attention import DifferentialAttention
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `differential_attention` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `attention`, `differential`, `noise-cancelling`, `rmsnorm`.

## Current model consumers

- [`wdformer`](../../wdformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
