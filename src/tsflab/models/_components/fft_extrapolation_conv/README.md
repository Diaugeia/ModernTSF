---
name: "fft_extrapolation_conv"
kind: "component"
module: "tsflab.models._components.fft_extrapolation_conv"
summary: "Zero-padded rfft convolution with per-bin complex weight and bias mapping a [B, C, L] history to a [B, C, H] horizon, with optional channel mixing of weight sets."
category: "frequency"
input: "x [batch, channels, input_length]; mixing [num_sets, channels] or None"
output: "[batch, channels, output_length]"
origin: "Independent Frequential Mapping of DiPE-Linear (Zhao et al., arXiv 2411.17257, Eqs. 3-6): zero-pad, rfft, complex multiply-accumulate per bin, irfft, keep the last H samples"
origin_models: ["dipelinear"]
tags: ["complex", "convolution", "fft", "frequency-domain", "horizon", "zero-padding", "stateless"]
---

# fft_extrapolation_conv

## Purpose

`FFTExtrapolationConv(input_length, output_length, num_sets)` forecasts a horizon
from a history with a single long circular convolution carried out in the
frequency domain. For `T = L + H - 1 + 2g` (guard `g = max(floor((L+H-1)/100), 1)`):

`y = irfft(theta * rfft(pad(x)) + beta, n=T)[-(H+g):-g]`

where `theta` and `beta` are complex vectors with `T // 2 + 1` entries, so each
frequency bin is processed independently (a kernel as long as the whole
history plus horizon, Eq. 6 of the paper). It replaces a dense `L x H` map by
`O(L + H)` parameters and an `O((L + H) log (L + H))` computation.

## Origin and granularity

Extracted from the DiPE-Linear implementation (the only consumer today). The
cut is exactly the pad, rfft, complex weighting, shift, irfft and slice; input
normalization, the attention maps applied before it, and weight-set routing stay
in the model. The guard band is an implementation detail of the official code,
not of the paper.

## Interface

`FFTExtrapolationConv(input_length, output_length, num_sets=1)`

- All sizes positive, else `ValueError`. Attributes `guard`, `padded_length`,
  `num_bins`.
- Parameters `real_weight`, `imag_weight`, `real_bias`, `imag_bias`, each
  `[num_sets, num_bins]` real tensors forming complex values (state-dict friendly).
  `real_weight[:, 0]` starts at 1 and everything else at 0 (an average filter).
- `forward(x, mixing=None)`: `x` is `[batch, channels, input_length]`. With
  `mixing=None` the layer needs `num_sets == 1` and all channels share the weight.
  With `mixing` of shape `[num_sets, channels]` channel `c` uses
  `sum_s mixing[s, c] * weight[s]` (and the same for the bias). Wrong shapes raise
  `ValueError`. Returns `[batch, channels, output_length]`.
- Stateless besides parameters.

## Invariants and equivalence evidence

- `tests/test_dipelinear.py` (FFTExtrapolationConvTests) checks equality with an
  explicit circular time-domain convolution of the padded history, the DC average
  initialization, and linear weight-set mixing; reference values in
  `tests/fixtures/components/fft_extrapolation_conv.pt`.
- `tests/test_dipelinear.py` (test_matches_official_reference_values)
  reproduces the official implementation's outputs through this layer
  (`tests/fixtures/dipelinear_reference.pt`).

## Variants and options

One weight set shared by all channels (`num_sets=1`), or `num_sets` sets mixed per
channel through an externally supplied matrix (see `weight_set_router`). One set
per channel is `num_sets=channels` with an identity mixing matrix.

## When to use and when not to use

Use for a parameter-efficient linear history-to-horizon map where frequency
independence is an acceptable inductive bias. Do not use when the horizon must
depend on cross-frequency interactions, for variable-length histories (the bin
count is fixed), or when a dense per-timestep map is required.

## Related components

`weight_set_router` (supplies `mixing`), `channel_wise_linear` (dense temporal
map), `wavelet`, `haar_dwt1d`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `FFTExtrapolationConv(input_length: int, output_length: int, num_sets: int=1)`
  Map ``(batch, channels, input_length)`` to a horizon by a long convolution.

```python
from tsflab.models._components.fft_extrapolation_conv import FFTExtrapolationConv
```

## Retrieval terms

`complex`, `convolution`, `fft`, `frequency-domain`, `horizon`, `zero-padding`

## Current model consumers (1)

`dipelinear`
<!-- component-card:generated:end -->
