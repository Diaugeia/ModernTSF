---
name: "frequency_band_sampler"
kind: "component"
module: "moderntsf.models._components.frequency_band_sampler"
summary: "Maps a layer index to a contiguous half-open FFT bin range (layer 0 highest frequencies, last layer near-DC) and slices a spectrum to it."
category: "frequency"
input: "spectrum complex [batch, freq, ...] (sample); num_bins and layer_idx integers (band)"
output: "band: (start, end) ints; sample: spectrum[:, start:end] with shape [batch, end - start, ...]"
origin: "Hierarchical frequency sampling of Dualformer, Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting (arXiv 2601.15669, 2026)"
origin_models: ["dualformer"]
tags: ["band", "depth", "fft", "frequency", "hierarchical", "sampling", "spectral"]
---

# frequency_band_sampler

## Purpose

`HierarchicalFrequencySampler(num_layers, alpha)` allocates a spectral band to
each network depth: shallow layers get high frequencies, deep layers get low
frequencies (layer `num_layers - 1` includes bin 0). For `n = num_bins`, `L =
num_layers`, layer `i`:

- tiling regime (`L == 1` or `alpha <= 1/L`): `start = int(n * (1 - (i+1)/L))`,
  `width = int(n / L)`.
- sliding regime (`alpha > 1/L`): `start = int(n * (1 - alpha) * (1 - i/(L-1)))`,
  `width = int(alpha * n)`; adjacent layers overlap.
- `end = min(n, start + max(width, 1))`; the range is `[start, end)`.

## Origin and granularity

Extracted from `dualformer` (commit `dd63af6c`, automated intake of SDMixer,
SEMixer, LSINet, Dualformer), where the frequency branch gives each encoder
layer its own band. The docstring describes it as the band arithmetic of
several dual time-frequency architectures; only Dualformer is recorded as the
source. Model-local in `dualformer`: the rfft/irfft around it (it builds a
band-limited signal by zeroing bins outside the range), the encoder layers, and
the fusion with the time branch.

## Interface

`HierarchicalFrequencySampler(num_layers: int, alpha: float = 1.0)`

- `num_layers` (int >= 1), `alpha` (float in `(0, 1]`); otherwise `ValueError`.
  Default `alpha=1.0` selects the sliding regime for `num_layers > 1`, i.e.
  full-width windows (every layer gets all bins).
- `band(num_bins, layer_idx) -> (start, end)`: `num_bins >= 1` and
  `0 <= layer_idx < num_layers`, else `ValueError`. Pure Python ints.
- `sample(spectrum, layer_idx) -> Tensor`: `spectrum.ndim >= 2` with the
  frequency axis at dim 1; returns `spectrum[:, start:end]` (a view, dtype and
  device preserved). `ValueError` for `ndim < 2`.
- No parameters, buffers, or state; attributes `num_layers`, `alpha`.
- CPU-confirmed examples for 17 bins: `L=3, alpha=0.3` gives
  `(11,16), (5,10), (0,5)`; `L=2, alpha=0.3` gives `(8,16), (0,8)`.

## Invariants and equivalence evidence

- no fixture and no dedicated unit test: nothing under `tests/` imports
  `HierarchicalFrequencySampler`. It is covered only indirectly by the
  `dualformer` model tests through the consumer.
- Integer truncation means the "non-overlapping" tiling is not exact: the
  bands can leave gaps (in the 17-bin example above, bin 10 and bin 16 belong
  to no layer). Layer 0 never includes the last bin in the tiling regime.

## Variants and options

`alpha` switches between tiling (`alpha <= 1/num_layers`) and overlapping
sliding windows (larger `alpha`). There is no log-spaced or learned variant.

## When to use and when not to use

Use to give each depth a fixed, deterministic frequency slice of a
`[batch, freq, ...]` spectrum (high to low). Do not use when bands must be
exactly partitioned or learned (see `freq_band_moe` for learned boundaries),
or when the frequency axis is not dim 1.

## Related components

`freq_band_moe` (gated band decomposition), `harmonic_energy_gate` (the other
Dualformer frequency component), `wavelet` (multi-resolution alternative).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `HierarchicalFrequencySampler(num_layers: int, alpha: float=1.0)`
  Assign a depth-indexed contiguous frequency band out of ``num_layers``.

```python
from moderntsf.models._components.frequency_band_sampler import HierarchicalFrequencySampler
```

## Retrieval terms

`band`, `depth`, `fft`, `frequency`, `hierarchical`, `sampling`, `spectral`

## Current model consumers (1)

`dualformer`
<!-- component-card:generated:end -->
