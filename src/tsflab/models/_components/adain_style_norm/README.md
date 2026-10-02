---
name: "adain_style_norm"
kind: "component"
module: "tsflab.models._components.adain_style_norm"
summary: "Adaptive instance normalization: standardize a [B, L, C] tensor over L with detached mean/std, then rescale to externally supplied style mean and std. No parameters."
category: "normalization"
input: "x [batch, length, channels]; style_mean [batch|1, 1, channels]; style_std [batch|1, 1, channels]"
output: "[batch, length, channels]"
origin: "Adaptive instance normalization (AdaIN; Huang and Belongie, ICCV 2017, image style transfer) used as the NSAN block of CANet (arXiv 2504.17913, 2025); extracted when CANet was added"
origin_models: ["canet"]
tags: ["adain", "adaptive", "non-stationary", "normalization", "style", "instance", "detach", "stateless", "parameter-free"]
---

# adain_style_norm

## Purpose

`AdaptiveInstanceNorm1d(eps)` standardizes a feature tensor along its sequence
axis and then re-injects caller-supplied statistics instead of learned affine
parameters. For `x` of shape `[B, L, C]`:

- `mu = mean_L(x)` and `sigma = std_L(x, unbiased=True) + eps`, both detached;
- `y = (x - mu) / sigma * style_std + style_mean`.

The module owns no parameters; the "style" is an input. Gradients flow through
`x` (as the numerator and through the constant `1/sigma` scale) and through
`style_mean` and `style_std`, but not through `mu` and `sigma`.

## Origin and granularity

The computation is the AdaIN operation of image style transfer (Huang and
Belongie, ICCV 2017), applied in CANet as its non-stationary adaptive
normalization (NSAN) after the spectral block, in place of a fixed LayerNorm
affine. It was added as a shared component in the CANet intake commit
(`b94ed873`, "add TQNet, TimePro, Gateformer, CANet"). The cut is only the
normalize-then-rescale arithmetic. Deriving the style statistics stays in
`canet`: its projection of the raw series mean/std, its `StyleBlendingGate`
(a fixed-ratio convex blend of two (mean, std) pairs followed by linear
projections, not a learned gate), and the style branch computed from patch
embeddings.

## Interface

`AdaptiveInstanceNorm1d(eps=1e-5)`

- `eps` (float): added to the standard deviation (not under the square root).
  No range validation is done; use a positive value.
- No parameters, buffers, or state-dict keys (`state_dict()` is empty).

`forward(x, style_mean, style_std)`

- `x`: floating tensor `[batch, length, channels]`. Raises `ValueError` if
  `x.ndim != 3`.
- `style_mean`, `style_std`: broadcastable to `[batch, 1, channels]` (a scalar
  tensor also broadcasts). They are not validated and are not detached.
  `style_std` is used as given (no eps, no positivity check).
- Returns a tensor with the broadcast shape of the result (the shape of `x`
  for the intended `[batch|1, 1, channels]` statistics) on the device and dtype
  of the inputs.
- Uses the unbiased standard deviation, so `length == 1` yields NaN (and a
  PyTorch warning). Normalization is always over axis 1.

## Invariants and equivalence evidence

- `tests/test_component_contracts_basic.py` (`test_adain_contract_and_reference`)
  checks the empty state dict, output shape and dtype, that output mean and
  unbiased std over the length axis equal the style statistics (loose tolerance
  because of eps), that `x`, `style_mean` and `style_std` all receive gradients,
  scalar-style broadcasting, and the `ValueError` for a 2-D input; the seeded
  output and `x` gradient are pinned by `tests/fixtures/components/adain_style_norm.pt`.
- `test_adain_rescales_normalized_features_to_style_statistics` in
  `tests/test_2025_query_gate_hyperstate_forecasters.py` repeats the statistics
  check on a shifted, scaled input with constant style statistics.
- The detached `mu`/`sigma` claim is by construction; no test isolates it. No
  pre-refactor fixture exists (the component was created together with its only
  consumer, `canet`), whose forward and backward are exercised by
  `test_canet_forward_and_gradient` in the same test file.

## Variants and options

None beyond `eps`. It is a different operation from `revin` (which uses its own
statistics and restores them later with an explicit denorm) and from
`last_value_center` (level-only centering, no scale).

## When to use and when not to use

Use when features should be standardized per sample over the sequence axis and
re-scaled to statistics produced elsewhere (for example per-sample series
moments or a learned style branch). Do not use when the statistics should be
the input's own and later undone (use `revin`), when the normalized axis is not
axis 1, when the standardization statistics should receive gradients, or for
inputs of length 1.

## Related components

`revin` (self-statistics, reversible), `last_value_center` (level-only
centering). The gate that blends the style statistics in `canet` is model-local
and is not `gated_fusion`: `gated_fusion` is a learned sigmoid gate over two
embeddings, whereas `canet` blends statistics with a fixed ratio.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `AdaptiveInstanceNorm1d(eps: float=1e-05)`
  Normalize over the sequence axis, then rescale to given statistics.

```python
from tsflab.models._components.adain_style_norm import AdaptiveInstanceNorm1d
```

## Retrieval terms

`adain`, `adaptive`, `non-stationary`, `normalization`, `style`

## Current model consumers (1)

`canet`
<!-- component-card:generated:end -->
