---
name: "gated_fusion"
kind: "component"
module: "tsflab.models._components.gated_fusion"
summary: "Convex blend of two equal-shaped tensors with a learned sigmoid gate: g = sigmoid(Wa a + Wb b), out = g * a + (1 - g) * b."
category: "fusion"
input: "a [..., dim]; b [..., dim] (identical shapes)"
output: "same shape as a and b"
origin: "Gated representation fusion of Gateformer (arXiv 2505.00307, 2025), generalized from its two uses; no earlier source recorded"
origin_models: ["gateformer"]
tags: ["fusion", "gate", "gated", "mixture", "sigmoid", "convex"]
---

# gated_fusion

## Purpose

`GatedFusion(dim)` computes an elementwise gate from both inputs and returns a
convex combination:

`gate = sigmoid(Linear_a(a) + Linear_b(b))`, `out = gate * a + (1 - gate) * b`.

Every output element lies between the corresponding elements of `a` and `b`.
Both linears are `dim -> dim` with bias, and the gate is per position and per feature.

## Origin and granularity

Added in commit `b94ed873` (automated intake of TQNet, TimePro, Gateformer, CANet)
for `gateformer`, which uses it twice: fusing a global (inverted) variate embedding
with a temporal-patch embedding, and fusing that result with a cross-variate encoder
output. The module docstring says several forecasters share this "global/local"
gate; only `gateformer` consumes it today, and no other consumer or pre-extraction
fixture exists. Cut at the fusion only: the branches feeding `a` and `b` and
their encoders remain model-local.

## Interface

`GatedFusion(dim: int)`

- `dim` (int >= 1): width of the last axis; `dim < 1` raises `ValueError`.
- `forward(a, b) -> Tensor`: `a` and `b` must have exactly equal shapes (any rank, last axis `dim`), else `ValueError`. Output has the same shape, dtype, and device. The usual call in `gateformer` is on `[B, n_vars, d_model]`.
- State-dict keys: `gate_a.weight`, `gate_a.bias`, `gate_b.weight`, `gate_b.bias` (each `[dim, dim]` or `[dim]`). Stateless, no dropout.
- The two inputs are not symmetric in parameters (`Wa` acts on `a`, `Wb` on `b`) but the blend is: the gate is the weight on `a`.

## Invariants and equivalence evidence

no fixture. `test_gated_fusion_interpolates_between_its_two_inputs` in
`tests/test_2025_query_gate_hyperstate_forecasters.py` sets the `a` gate bias to 10
and `Wb` to zero and checks the output approaches `a`. The `gateformer` model tests in
the same file cover the composed forward and backward pass.

## Variants and options

Only `dim`. Not covered: gates of different widths or inputs, scalar (per-position)
gates, a residual form `a + g * b`, or gating on concatenated inputs.

## When to use and when not to use

Use when two branches produce same-shaped embeddings and a data-dependent convex
blend per feature is wanted. Do not use when the branches have different shapes
(project them first), when the fusion should be additive or concatenative, or when
the gate should be shared across features.

## Related components

`mixer_block` (residual mixing within one stream), `flatten_forecast_head` (used by
`gateformer` to produce the temporal branch).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `GatedFusion(dim: int)`
  Blend two same-shaped embeddings with a learned per-position gate.

```python
from tsflab.models._components.gated_fusion import GatedFusion
```

## Retrieval terms

`fusion`, `gate`, `gated`, `mixture`, `sigmoid`

## Current model consumers (1)

`gateformer`
<!-- component-card:generated:end -->
