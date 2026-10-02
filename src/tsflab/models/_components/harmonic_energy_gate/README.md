---
name: "harmonic_energy_gate"
kind: "component"
module: "tsflab.models._components.harmonic_energy_gate"
summary: "Per-channel share of spectral energy carried by the dominant low-frequency fundamental and its first harmonics, used as a periodicity gate in [0, 1]."
category: "fusion"
input: "x [batch, length, channels]"
output: "[batch, 1, channels], values in [0, 1]"
origin: "Harmonic-energy weighting of Dualformer, Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting (arXiv 2601.15669, 2026)"
origin_models: ["dualformer"]
tags: ["energy", "fusion", "gate", "harmonic", "periodicity", "spectral", "weighting", "non-differentiable-selection"]
---

# harmonic_energy_gate

## Purpose

`HarmonicEnergyGate(num_harmonics=3, low_freq_guard=3)` measures how periodic
each channel is. With `A = |rfft(x - mean_t x)|` (`nb = length // 2 + 1` bins):

- candidates are bins in `[low_freq_guard, max(low_freq_guard, nb // num_harmonics))`;
  `f0 = argmax` of `A` over that range, per sample and channel;
- `E_h = sum_{k=1..num_harmonics} A[min(k * f0, nb - 1)]^2`;
- output `E_h / (sum_f A[f]^2 + 1e-5)`.

A channel whose energy sits on one fundamental and its harmonics scores near
1; diffuse spectra score near 0.

## Origin and granularity

Extracted from `dualformer` (commit `dd63af6c`, automated intake). The
consumer applies it to the embedded sequence `[B, L, d_model]` and mixes
`freq_state * w + time_state * (1 - w)`; the code comment calls this the paper's
periodicity-aware gate. Cut at the ratio itself; the two branches, the fusion
rule, and the choice of what to feed it remain in `dualformer`. The module is
parameter-free.

## Interface

`HarmonicEnergyGate(num_harmonics: int = 3, low_freq_guard: int = 3)`

- `num_harmonics` (int >= 1): fundamental plus the next `num_harmonics - 1`
  multiples. `low_freq_guard` (int >= 1): bins below this index (including DC)
  are never the fundamental; `dualformer` passes its `dc_bins` here.
- `forward(x)`: floating `[batch, length, channels]`. Raises `ValueError` for
  `ndim != 3` and when `length // 2 + 1 <= low_freq_guard`. Returns
  `[batch, 1, channels]` on the same device and real dtype as `x`.
- No parameters, buffers, or state. The argmax selection is not differentiable;
  gradient flows only through the amplitudes at the selected bins and the total.
- Quirk: when `nb // num_harmonics <= low_freq_guard` the candidate range is
  empty, `f0` falls back to bin 0, and the output is about 0 (DC removed), so
  short inputs or many harmonics silently yield a zero gate.

## Invariants and equivalence evidence

- The `dualformer` model tests also exercise it indirectly.
- CPU check: a pure sinusoid with 4 cycles over length 32 returns exactly 1.0.
- The ratio is at most 1 up to the `1e-5` stabilizer, because harmonics are
  confined to bins below `nb` and counted once each unless clamped.
- `tests/test_component_contracts_signal.py` pins the interface (shapes, dtype, state-dict keys, invariants, gradient flow, error cases) and a seeded numerical regression against `tests/fixtures/components/harmonic_energy_gate.pt`.

## Variants and options

`num_harmonics` and `low_freq_guard` only. The ratio uses the sample's own
spectrum (no running statistics), so it is recomputed per forward call.

## When to use and when not to use

Use to weight a frequency-domain branch against a time-domain branch per
channel from a `[B, L, C]` tensor. Do not use for very short sequences relative
to `low_freq_guard * num_harmonics`, for signals with aperiodic structure where
the low-frequency fundamental is ambiguous, or when a learned gate is needed
(this one has no parameters).

## Related components

`dominant_periods` (explicit period discovery), `gated_fusion` (learned fusion
gates), `frequency_band_sampler` (the other Dualformer frequency component).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `HarmonicEnergyGate(num_harmonics: int=3, low_freq_guard: int=3)`
  Return each channel's harmonic-energy share of its total spectral energy.

```python
from tsflab.models._components.harmonic_energy_gate import HarmonicEnergyGate
```

## Retrieval terms

`energy`, `fusion`, `gate`, `harmonic`, `periodicity`, `spectral`, `weighting`

## Current model consumers (1)

`dualformer`
<!-- component-card:generated:end -->
