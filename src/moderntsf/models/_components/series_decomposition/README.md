---
name: "series_decomposition"
kind: "component"
module: "moderntsf.models._components.series_decomposition"
summary: "Edge-padded moving average over time and the residual/trend split x = (x - MA(x)) + MA(x) for [B, L, C] series."
category: "decomposition"
input: "[batch, length, channels]"
output: "EdgePaddedMovingAverage: [batch, length, channels] (stride 1); SeriesDecomposition: (residual, trend), each [batch, length, channels]"
origin: "Moving-average series decomposition block of Autoformer, Wu et al., NeurIPS 2021, reused by DLinear (AAAI 2023) and others"
origin_models: ["autoformer", "dlinear", "fedformer"]
tags: ["decomposition", "moving-average", "residual", "smoothing", "trend", "padding", "stateless"]
---

# series_decomposition

## Purpose

`EdgePaddedMovingAverage(kernel_size, stride)` smooths each channel over time
with a window of `k = kernel_size` after repeating the first and last time step
`(k - 1) / 2` times at the two ends, so with stride 1 the output keeps the
length. `SeriesDecomposition(kernel_size)` uses it with stride 1:

`trend = MA(x)`, `residual = x - trend`, returned as `(residual, trend)`, so
`residual + trend == x`. Callers often call the residual "seasonal".

## Origin and granularity

This is the `series_decomp` / `moving_avg` block of Autoformer, also used in
FEDformer and DLinear. It was extracted in commit `15bc27e6` ("extract series
decomposition") from near-identical local copies in `amplifier`, `bist`,
`moderntcn`, `stop`, and the shared `autoformer_encdec` and `dlinear`
components; `15bc27e6`'s follow-up `b1518394` migrated `timemixer`. The cut
is the moving average plus subtraction. Model-local: multi-kernel or
learned decompositions (for example MICN-style several kernels are just several
instances), DUET's per-expert moving average (kept local on purpose, because
it supports even kernels through asymmetric padding while this component
rejects them, which `bist` relies on), and what is done with the two parts.
Current consumers: `autoformer`, `fedformer`, `micn`, `symtime`, `refocus`,
`amplifier`, `stop`, `timemixer`, `moderntcn`, `bist`, plus the `dlinear`
component.

## Interface

`EdgePaddedMovingAverage(kernel_size, stride=1)`

- `kernel_size` (odd int >= 1): window; even or < 1 raises `ValueError`
  ("kernel_size must be a positive odd integer"). `stride` (int >= 1): raises
  `ValueError` otherwise.
- `forward(x)`: `x` `[batch, length, channels]` (rank 3 required by the
  indexing). Output length is `length` when `stride == 1`, and
  `floor((length + k - 1 - k) / stride) + 1` otherwise, so with `stride > 1` it
  no longer matches the input. `kernel_size > length` works (edge repeats).
- Attributes `kernel_size`, `avg` (`nn.AvgPool1d`).

`SeriesDecomposition(kernel_size)`

- `forward(x)` returns `(residual, trend)` as two tensors of the shape of `x`.
  Attribute `moving_avg`.
- Both modules have no parameters or buffers (empty state dict), are
  differentiable, and keep dtype/device. They are stateless.

## Invariants and equivalence evidence

- `test_edge_padded_series_decomposition_matches_reference` in
  `tests/test_repository_contracts.py`: output equals explicit edge padding plus
  `avg_pool1d`, `residual + trend == x`, gradients finite, even kernel
  rejected.
- `test_autoformer_decomposition_and_fft_delay_equations` in
  `tests/test_transformer_patch_forecasters_a.py` checks the hand-computed trend
  `[5/3, 3, 13/3]` for `[1, 3, 5]` with kernel 3.
- `tests/fixtures/component_extraction_batch7.pt` with
  `tests/test_component_extraction_batch7.py` pins the migrated `timemixer`
  (outputs, state dict, gradients) against pre-refactor values.
- `tests/test_component_extraction_moe.py` documents (and tests) why DUET's
  moving average was not extracted.

## Variants and options

- `EdgePaddedMovingAverage` alone for smoothing only (`refocus`, `timemixer`).
- `stride > 1` downsamples the smoothed series; the decomposition class always
  uses stride 1.
- Not provided: even kernels, learnable kernels, zero or reflect padding, and
  decompositions by frequency (see `haar_dwt1d`, `wavelet`).

## When to use and when not to use

Use for Autoformer-style trend/residual splitting of `[B, L, C]` tensors with an
odd window. Do not use when an even window or different boundary handling is
specified by the paper, when the time axis is not axis 1, or when the trend
must be learned.

## Related components

`dlinear` (composes it with two projections), `channel_wise_linear`,
`haar_dwt1d` and `wavelet` (frequency-based splits), `revin`,
`last_value_center`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `EdgePaddedMovingAverage(kernel_size: int, stride: int=1)`
  Smooth a series after repeating its first and last observations.
- `SeriesDecomposition(kernel_size: int)`
  Split ``(B, L, C)`` values into residual and moving-average trend.

```python
from moderntsf.models._components.series_decomposition import EdgePaddedMovingAverage, SeriesDecomposition
```

## Retrieval terms

`decomposition`, `moving-average`, `residual`, `smoothing`, `trend`

## Current model consumers (10)

`amplifier`, `autoformer`, `bist`, `fedformer`, `micn`, `moderntcn`, `refocus`, `stop`, `symtime`, `timemixer`
<!-- component-card:generated:end -->
