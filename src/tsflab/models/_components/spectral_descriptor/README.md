---
name: "spectral_descriptor"
kind: "component"
module: "tsflab.models._components.spectral_descriptor"
summary: "Parameter-free window statistics: normalized spectral entropy and low/mid/high band-energy ratios of the channel-averaged power spectrum."
category: "frequency"
input: "x [batch, length, channels]"
output: "[batch, 4] = (entropy, low, mid, high), each in [0, 1]"
origin: "Spectral descriptor of CoRe, Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting (arXiv 2609.34638, 2026), Eqs. 10-17"
origin_models: ["core"]
tags: ["band", "descriptor", "energy", "entropy", "fft", "frequency", "power", "ratio", "spectral", "spectrum", "gate-input"]
---

# spectral_descriptor

## Purpose

`SpectralDescriptor(eps=1e-8)` summarizes how a window's spectral energy is
distributed. Per sample, with each channel's temporal mean removed:

- `P = mean_c |rfft(x)|^2` over `K = length // 2 + 1` bins;
- `p = (P + eps) / (sum P + K * eps)`;
- entropy `H = -sum p log(p + eps) / log(max(K, 2))`;
- band ratios: sums of `p` over bins `[0, K//3]`, `[K//3 + 1, 2K//3]`,
  `[2K//3 + 1, K - 1]`.

It returns `[H, low, mid, high]`; the three ratios sum to one. Low entropy
means energy is concentrated in few frequencies; the band ratios say where.

## Origin and granularity

Extracted from `core` (CoRe, automated intake test). CoRe feeds the descriptor
into a gate that scales its cross-variate correction. The cut is at the
statistic itself: the gate network, the correction branch, and how the gate is
applied stay in `core`. The existing `harmonic_energy_gate` and
`frequency_band_sampler` were retrieved first and rejected: they return a
per-channel periodicity ratio and sampled band indices, not window-level
entropy plus band ratios.

## Interface

`SpectralDescriptor(eps: float = 1e-8)`

- `eps` (float > 0): smoothing added to every bin; raises `ValueError` otherwise.
- `forward(x)`: floating `[batch, length, channels]`; raises `ValueError` when
  `x.ndim != 3`. Returns `[batch, 4]` on the device and real dtype of `x`.
- No parameters, buffers, or state; fully differentiable (no argmax).
- Channels are averaged before normalization, so the output describes the
  window, not each channel.

## Invariants and equivalence evidence

- `tests/test_core_tta.py` compares the module with an independent float64
  oracle over several lengths, and checks the entropy range and that the band
  ratios sum to one.
- Differences from the official CoRe code are recorded in the `core` card:
  statistics are per sample (official: per batch) and band edges follow the
  paper's inclusive ranges (official: half-open thirds).

## Variants and options

`eps` only. Band edges are fixed thirds of the one-sided spectrum; a different
partition belongs in a new option, not in a consumer-side edit.

## When to use and when not to use

Use as a cheap, parameter-free conditioning signal for gates, routers, or
schedulers that should react to how periodic or noisy a window is. Do not use
when per-channel spectra matter (channels are averaged), for very short windows
(few bins make the thirds coarse), or when the band split must follow known
physical frequencies.

## Related components

`harmonic_energy_gate` (per-channel periodicity ratio), `frequency_band_sampler`
(band index sampling), `dominant_periods` (explicit period discovery),
`energy_frequency_pooling`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `SpectralDescriptor(eps: float=1e-08)`
  Map ``[batch, length, channels]`` to ``[batch, 4]`` spectral statistics.

```python
from tsflab.models._components.spectral_descriptor import SpectralDescriptor
```

## Retrieval terms

`band`, `descriptor`, `energy`, `entropy`, `fft`, `power`, `ratio`, `spectral`, `spectrum`

## Current model consumers (1)

`core`
<!-- component-card:generated:end -->
