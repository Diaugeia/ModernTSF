---
name: "softmax_gate"
kind: "component"
module: "tsflab.models._components.softmax_gate"
summary: "Softmax feature gate: out = x * softmax(Linear(x), dim=-1) over the last axis, no residual or normalization."
category: "mixer"
input: "[..., dim]"
output: "same shape as input"
origin: "Gated attention block of TSMixer/PatchTSMixer, Ekambaram et al., KDD 2023 (arXiv 2306.09364), Sec. 3.3.5 and Appendix Fig. 6b"
origin_models: ["patchtsmixer"]
tags: ["gate", "gated-attention", "softmax", "feature", "mixer", "stateless"]
---

# softmax_gate

## Purpose

`SoftmaxGate(dim)` computes `w = softmax(Linear(x), dim=-1)` and returns `x * w`
(elementwise). Because `w` sums to one over the feature axis, it acts as a
feature-importance filter: dominant features are kept, the rest are damped. The
paper calls this "gated attention" although it has no query/key interaction.

## Origin and granularity

Extracted with PatchTSMixer, which applies it three times (after the patch-mixing
MLP, the feature-mixing MLP, and the channel-mixing MLP, each on a different
permuted axis). Only the gate is shared; the MLP, norm, and residual stay in the
model. Distinct from `gated_fusion` (sigmoid convex blend of two tensors) and
`gated_dilated_conv` (tanh*sigmoid WaveNet unit).

## Interface

`SoftmaxGate(dim: int)`

- `dim` (int >= 1): width of the last axis; `dim < 1` raises `ValueError`.
- `forward(x [..., dim]) -> Tensor`: same shape, dtype, device. A last axis other than `dim` raises `ValueError`.
- State-dict keys: `score.weight` `[dim, dim]`, `score.bias` `[dim]`. No dropout; stateless apart from parameters.

## Invariants and equivalence evidence

- Gate weights are in (0, 1) and sum to one over the last axis, so `|out| <= |x|` elementwise.
- Contract, invariant, gradient, and seeded regression tests: `tests/test_component_softmax_gate.py`, reference values in `tests/fixtures/components/softmax_gate.pt`.
- Matches the Hugging Face PatchTSMixer gated-attention module (`Linear` then `Softmax(dim=-1)` then product) in the model tests of `tests/test_patchtsmixer_structure.py`.

## Variants and options

Only `dim`. Not covered: gates whose output width differs from the input, sigmoid gates, temperature, or a residual form.

## When to use and when not to use

Use to filter the features of a tensor by a learned softmax distribution along its last axis, typically after an MLP and before a residual add. Do not use for a two-input blend (`gated_fusion`), for a gate that must not sum to one across features, or when the gated axis is not last (permute first).

## Related components

`gated_fusion`, `gated_dilated_conv`, `mixer_block` (the TSMixer block, which has no gate).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `SoftmaxGate(dim: int)`
  Multiply the last axis by ``softmax(Linear(x), dim=-1)``.

```python
from tsflab.models._components.softmax_gate import SoftmaxGate
```

## Retrieval terms

`gate`, `gated-attention`, `softmax`, `feature`, `mixer`

## Current model consumers (1)

`patchtsmixer`
<!-- component-card:generated:end -->
