---
name: "fft_extrapolation_conv"
kind: "component"
module: "tsflab.models._components.fft_extrapolation_conv"
summary: "Zero-padded rfft convolution with per-bin complex weight and bias mapping a [B, C, L] history to a [B, C, H] horizon, with optional channel mixing of weight sets."
category: "frequency"
input: "x [batch, channels, input_length]; mixing [num_sets, channels] or None"
output: "[batch, channels, output_length]"
origin: "Independent Frequential Mapping of DiPE-Linear (Zhao et al., arXiv 2411.17257): zero-pad, rfft, complex multiply-accumulate per bin, irfft, keep the last H samples"
origin_models: ["dipelinear"]
tags: ["complex", "convolution", "fft", "frequency-domain", "horizon", "zero-padding", "stateless", "linear", "parameter-efficient", "forecast-head"]
---

# fft_extrapolation_conv

## Purpose

`FFTExtrapolationConv(input_length, output_length, num_sets)` forecasts a horizon
from a history with a single long circular convolution carried out in the
frequency domain. With `L = input_length`, `H = output_length`,
`g = max(floor((L + H - 1) / 100), 1)` and `T = L + H - 1 + 2g`:

`y = irfft(theta * rfft(pad(x)) + beta, n=T)[..., -(H + g):-g]`

where `pad` adds `g` zeros on the left and `H - 1 + g` zeros on the right, and
`theta` and `beta` are complex vectors with `T // 2 + 1` entries, so each
frequency bin is processed independently (a circular kernel as long as the whole
padded history plus horizon). It replaces a dense `L x H` map by
`O(L + H)` parameters and an `O((L + H) log (L + H))` computation. Note `beta`
is added in the frequency domain after the product, so it is a (circular) time
offset signal, not a per-bin scale.

## Origin and granularity

Extracted from the DiPE-Linear implementation (the only consumer today; see the
`dipelinear` card for the paper equation numbers). The cut is exactly the pad,
rfft, complex weighting, shift, irfft and slice; input normalization, the
attention maps applied before it, and the loss stay in the model, and the
weight-set routing matrix comes from `weight_set_router`. The guard band is an
implementation detail of the official code, not of the paper.

## Interface

`FFTExtrapolationConv(input_length, output_length, num_sets=1)`

- All three sizes must be >= 1, else `ValueError`. Plain attributes
  `input_length`, `output_length`, `num_sets`, `guard`, `padded_length`,
  `num_bins`.
- Parameters `real_weight`, `imag_weight`, `real_bias`, `imag_bias`, each
  `[num_sets, num_bins]` real tensors combined into complex values on the fly
  (state-dict friendly). `real_weight[:, 0]` starts at 1 and everything else at
  0, so the initial layer outputs the same constant for every horizon step:
  `sum(x) / padded_length` (the mean of the zero-padded history, not of `x`).
- `forward(x, mixing=None)`: `x` must be `[batch, channels, input_length]`
  (else `ValueError`). With `mixing=None` the layer needs `num_sets == 1` (else
  `ValueError`) and all channels share the weight. With `mixing` of shape
  `[num_sets, channels]` channel `c` uses `sum_s mixing[s, c] * weight[s]` (and
  the same for the bias, computed by `weight_set_router.mix_weight_sets`, which
  has identical semantics to the former inline einsum); any other shape raises `ValueError`. `mixing` is cast
  to complex, and may carry gradients. Returns `[batch, channels, output_length]`
  with the real dtype and device of `x` (float32 or float64; `rfft` support for
  other dtypes is backend dependent).
- Fixed lengths: the bin count depends on `input_length`, so another history
  length is rejected. Stateless besides parameters (no buffers, no randomness).

## Invariants and equivalence evidence

- `tests/test_dipelinear.py` (`FFTExtrapolationConvTests`) checks, on random
  parameters, equality with an explicit circular time-domain convolution of the
  padded history plus the inverse-transformed bias; the DC average
  initialization; and linear weight-set mixing (one-hot mixing equals running a
  single-set layer on each channel group) with the `ValueError` cases for missing
  or wrongly shaped `mixing`. The mixing test also pins its output in
  `tests/fixtures/components/fft_extrapolation_conv.pt`.
- `test_matches_official_reference_values` in the same file reproduces the
  official DiPE-Linear implementation's forecasts through this layer
  (`tests/fixtures/dipelinear_reference.pt`; single expert, three experts, and
  history shorter and longer than the horizon).
- Not tested directly: gradients of the layer alone (the `dipelinear` training
  tests cover them indirectly), float64 inputs, and the constructor `ValueError`.

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

`weight_set_router` (supplies `mixing`; its `mix_weight_sets` computes the same
per-channel combination for real weights, which this layer repeats inline for
complex ones), `channel_wise_linear` and `flatten_forecast_head` (dense,
time-domain history-to-horizon maps; this layer is the frequency-domain
parameter-efficient alternative), `dlinear` (dense linear baseline backbone).
`wavelet` and `haar_dwt1d` are fixed (not learned) multi-band transforms and are
not alternatives to this map.

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
