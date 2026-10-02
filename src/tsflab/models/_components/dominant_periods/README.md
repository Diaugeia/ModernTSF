---
name: "dominant_periods"
kind: "component"
module: "tsflab.models._components.dominant_periods"
summary: "Top-k FFT period selection: integer periods T // f from the batch-mean amplitude spectrum, plus per-sample amplitudes at those frequencies."
category: "frequency"
input: "x [batch, time, channels]"
output: "periods int64 ndarray [k] (shared by the batch, not a tensor); amplitudes [batch, k]"
origin: "Period discovery step of TimesNet, Wu et al., ICLR 2023 (TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis); the same routine is reused by MSGNet (AAAI 2024)"
origin_models: ["timesnet", "msgnet"]
tags: ["amplitude", "fft", "frequency", "period", "spectrum", "top-k", "numpy-output"]
---

# dominant_periods

## Purpose

`dominant_periods(x, k=2)` finds the `k` strongest non-DC frequencies of a
batch of series and converts them to integer periods. With
`S = rfft(x, dim=1)` (shape `[B, T//2+1, C]`):

- `strength[f] = mean_b mean_c |S[b, f, c]|`, with `strength[0] = 0` (DC removed).
- `idx = topk(strength, k)` (descending strength); `periods = T // idx`.
- `amplitudes[b, j] = mean_c |S[b, idx[j], c]|`.

The periods are one set shared across the whole batch; the amplitudes are
per-sample and keep the autograd graph, the periods do not.

## Origin and granularity

The routine is the period-discovery step of TimesNet (FFT_for_Period) and is
reused by `msgnet` for its scale-graph branches. The git history does not
record the first extraction commit of this component (the file was moved
during the colocation refactor `33ea2050`); the origin is taken from its two
consumers and from the test name. It is cut at the pure function: spectrum,
top-k, period arithmetic. Model-local: how periods are used (TimesNet folds the
series into a 2D image of shape `[B, ceil(T/p), p, C]` and applies an Inception
block; MSGNet builds one graph branch per period), and the weighting of branch
outputs (softmax over the amplitudes in both consumers).

## Interface

`dominant_periods(x: Tensor, k: int = 2) -> tuple[np.ndarray, Tensor]`

- `x`: floating tensor `[batch, time, channels]`; `ndim != 3` raises `ValueError`.
- `k`: int in `[1, T//2]` (number of non-DC bins); otherwise `ValueError`.
- Returns `(periods, amplitudes)`: `periods` is a numpy `int64` array of shape
  `[k]` (computed on CPU via `.detach().cpu().numpy()`, so each call forces a
  device sync); `amplitudes` is a tensor `[batch, k]` on the input device with
  gradient to `x`.
- No parameters, buffers, or state. The `rfft` is computed twice internally.
- Edge case: if the batch-mean spectrum is all zero (for example a constant
  zero input after DC removal) `topk` may return index 0, which gives a numpy
  divide-by-zero warning and a period of 0; callers must guard `period >= 1`
  (MSGNet clamps with `max(1, ...)`; TimesNet does not).

## Invariants and equivalence evidence

- Contract, invariant, gradient, and seeded numerical-regression tests: `tests/test_component_contracts_basic.py`, reference values in `tests/fixtures/components/dominant_periods.pt`.
- `test_dominant_periods_matches_timesnet_msgnet_reference` in
  `tests/test_repository_contracts.py` checks periods, amplitudes, and the
  gradient of the amplitudes against an inline reference implementation.
- `test_timesnet_period_discovery_equation` in
  `tests/test_transformer_patch_forecasters_a.py` checks that a sine of period 4
  over length 16 yields period `[4]` and amplitudes of shape `[1, 1]`.
- no fixture: there is no stored pre-refactor tensor fixture; the inline
  reference above is the equivalence evidence.

## Variants and options

Only `k`. There is no option for per-sample periods, windowing, or
interpolation of fractional periods: the period is the integer quotient
`T // f`, so non-divisor frequencies are rounded down.

## When to use and when not to use

Use for multi-period models that fold or group a `[B, T, C]` window by a small
number of dominant cycles. Do not use when periods must differ per sample (they
are batch-shared), when the whole call must stay on-device (it returns a numpy
array), or when a frequency-domain gate is needed instead of integer periods
(see `harmonic_energy_gate`).

## Related components

`harmonic_energy_gate` (periodicity measured as spectral energy ratio),
`frequency_band_sampler`, `periodic_query_bank` (a fixed known period instead
of a discovered one).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `dominant_periods(x: torch.Tensor, k: int=2)`
  Return top-k integer periods and per-sample FFT amplitudes for BLC data.

```python
from tsflab.models._components.dominant_periods import dominant_periods
```

## Retrieval terms

`amplitude`, `fft`, `frequency`, `period`, `spectrum`

## Current model consumers (2)

`msgnet`, `timesnet`
<!-- component-card:generated:end -->
