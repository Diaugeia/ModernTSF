---
name: "haar_dwt1d"
kind: "component"
module: "moderntsf.models._components.haar_dwt1d"
summary: "Single-level orthonormal Haar DWT along the last axis returning (approx, detail) of length ceil(T/2), with an exact inverse that truncates odd-length padding."
category: "decomposition"
input: "x [..., T] (T >= 2); inverse: approx [..., M], detail [..., M]"
output: "forward: (approx, detail) each [..., ceil(T/2)]; inverse: [..., 2M] or [..., length]"
origin: "Haar wavelet sub-series mapping of SWIFT, Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting (arXiv 2501.16178, 2025)"
origin_models: ["swift"]
tags: ["dwt", "haar", "sub-series", "wavelet", "lossless", "orthonormal", "odd-length-padding"]
---

# haar_dwt1d

## Purpose

`HaarDWT1D` rotates adjacent sample pairs `(a, b)` into
`approx = (a + b) / sqrt(2)` and `detail = (a - b) / sqrt(2)`.
`HaarIDWT1D` inverts it: `even = (approx + detail) / sqrt(2)`,
`odd = (approx - detail) / sqrt(2)`, interleaved. The transform is orthonormal
and lossless; an odd-length input is first extended by replicating its last
sample, and the inverse can truncate back with `length`.

## Origin and granularity

Extracted from `swift` (commit `6b491e13`, automated intake), where the
forward transform splits the (reversible-normalized) window into sub-series, a
small convolution fuses them, per-sub-series linear maps predict the output
sub-series, and the inverse reconstructs the horizon. Only the Haar pair is
here; the fusion convolution, the linear maps, and the `_half_length`
bookkeeping stay in `swift`. For multi-level or longer filters use `wavelet`.

## Interface

`HaarDWT1D()` and `HaarIDWT1D()`, neither takes constructor arguments.

- `HaarDWT1D.forward(x) -> (approx, detail)`: `x` floating, any leading shape,
  last axis `T >= 2` (else `ValueError`). Both outputs have last length
  `ceil(T / 2)`; leading axes, dtype, and device are preserved.
- `HaarIDWT1D.forward(approx, detail, length=None) -> Tensor`: shapes must match
  (else `ValueError`); output last length `2 * approx.shape[-1]`, then truncated
  to `length` when given (pass the original `T` for odd inputs).
- No parameters, buffers, or state; no stored padding record, so the caller must
  remember the original length.

## Invariants and equivalence evidence

- `test_haar_dwt_round_trip_is_lossless_for_even_and_odd_lengths` in
  `tests/test_frequency_wavelet_attention_forecasters.py` checks reconstruction
  for lengths 8 and 9 on `[2, 3, T]` tensors.
- `test_haar_dwt_matches_closed_form_on_a_known_pair` in the same file checks
  `[2, 4, 6, 8]` against the closed form.
- no fixture: no stored pre-refactor tensors; both tests above are closed-form.

## Variants and options

None. Single level, Haar only, last axis only, no boundary modes. For
multi-level, db2/db4, or undecimated transforms see `wavelet`, whose
`DecimatedWaveletTransform("haar")` is a stateful alternative that keeps the
trim record internally.

## When to use and when not to use

Use for a stateless, exact one-level Haar split of `[..., T]` tensors where the
caller tracks the original length. Do not use for deeper pyramids or
non-Haar filters (use `wavelet`), or when sub-series must stay aligned
with the time axis (the outputs are decimated by 2).

## Related components

`wavelet` (multi-level, db2/db4, a-trous), `series_decomposition` (moving-average
trend/seasonal split), `revin` (applied before the DWT in `swift`).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `HaarDWT1D()`
  Forward single-level Haar DWT along the last axis.
- `HaarIDWT1D()`
  Inverse single-level Haar DWT along the last axis.

```python
from moderntsf.models._components.haar_dwt1d import HaarDWT1D, HaarIDWT1D
```

## Retrieval terms

`dwt`, `haar`, `sub-series`, `wavelet`

## Current model consumers (1)

`swift`
<!-- component-card:generated:end -->
