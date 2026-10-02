---
name: "wavelet"
kind: "component"
module: "tsflab.models._components.wavelet"
summary: "Fixed-filter wavelet transforms on [batch, channels, length]: multi-level decimated DWT (haar/db1/db2/db4, exact inverse for haar) and a-trous undecimated stationary transform."
category: "decomposition"
input: "x [batch, channels, length]"
output: "list of subband tensors [approx_L, detail_L, ..., detail_1]; decimated lengths shrink per level, undecimated keep length"
origin: "Fixed Daubechies filter banks used by clean-room wavelet forecasters: WDformer (arXiv 2509.25231), DPWMixer (arXiv 2512.02070), AWEMixer (arXiv 2511.04722), all 2025 arXiv preprints; the transforms are standard wavelet analysis"
origin_models: ["wdformer", "dpwmixer", "awemixer"]
tags: ["dwt", "haar", "multi-resolution", "subband", "undecimated", "wavelet", "a-trous", "daubechies", "stateful"]
---

# wavelet

## Purpose

Two channel-independent (grouped, per-channel) wavelet analysis modules over
`[B, C, L]` with fixed filters (`haar`, `db1`, `db2`, `db4`):

- `DecimatedWaveletTransform.decompose`: stride-2 filtering per level on the
  running approximation, returning `[approx_L, detail_L, ..., detail_1]`
  (coarsest first, the `pywt.wavedec` order).
- `UndecimatedWaveletTransform`: a-trous stationary transform, dilation `2**i`
  at level `i`, stride 1, so every subband has length `L`; returns the same
  coarsest-first list `[approx_L, detail_L, ..., detail_1]`.

The filter taps are entered as constants (closed form for haar/db2, tabulated
for db4), not imported from a wavelet package.

## Origin and granularity

Created in commit `d73cf9d0` (automated intake of DPWMixer, AWEMixer, TimeExpert,
WDformer) as the shared wavelet layer. Consumers: `wdformer` (haar decimated,
`level=wave_size`, one instance to embed and one to reconstruct the forecast with
`reconstruct`), `dpwmixer` (haar `level=1`, keeping only the approximation to
build a downsampling pyramid), `awemixer` (undecimated; db4 level 3 are the defaults in
both the component and the model; subbands feed an adaptive router).
The cut is the filter bank and its inverse; learned mixing, routing, and
per-subband heads stay in the models. The single-level stateless Haar pair is a
separate component, `haar_dwt1d` (from `swift`). `WPMixer` and `sonnet`
define wavelet code locally and do not use this component.

## Interface

`available_wavelets() -> tuple[str, ...]` returns the sorted names
`('db1', 'db2', 'db4', 'haar')`.

`DecimatedWaveletTransform(wavelet: str = "haar", level: int = 1)`

- `wavelet` must be in `available_wavelets()` (else `ValueError`); `level >= 1`
  (else `ValueError`).
- Buffers `low`, `high`, each `[1, 1, filter_len]`, float32 (they appear in the
  state dict and follow `.to(device/dtype)`). No parameters.
- `decompose(x)`: odd lengths are replicate-padded by one sample per level;
  filters longer than 2 taps additionally use circular padding of
  `filter_len - 2` each side. Haar lengths follow `ceil(L / 2)` per level; longer
  filters give longer subbands (CPU-confirmed: db4 at `L=32` level 1 gives 19
  samples, not 16). It records per-level odd-padding flags in `self._trims`.
- `reconstruct(coeffs)`: only for two-tap filters (haar/db1), else
  `NotImplementedError`; `len(coeffs)` must be `level + 1` (else `ValueError`).
  It trims by the flags stored from the most recent `decompose` call, so it is
  stateful: reconstruct a signal only after decomposing one with the same
  odd/even length pattern on this instance (CPU-confirmed round-trip error
  about 1e-7 for haar level 2 at `L=10`). Using one instance for different
  lengths in alternation corrupts the trims; use separate instances (as
  `wdformer` does for embed and output).

`UndecimatedWaveletTransform(wavelet: str = "db4", level: int = 3)`

- Same validation and buffers. `forward(x)` returns the list described above, all
  `[B, C, L]`. Circular, left-only padding of `(filter_len - 1) * 2**i`; this is
  causal-style with wraparound, not the symmetric a-trous filter. If the pad
  exceeds `L` torch raises `RuntimeError` (CPU-confirmed: db4 level 3 at `L=10`),
  so require `L >= (filter_len - 1) * 2**(level - 1)`. No inverse is provided.

## Invariants and equivalence evidence

- `tests/test_wdformer_structure.py`, `tests/test_dpwmixer_structure.py` (a haar
  decomposition on `[2, 4, 12]`) and `tests/test_awemixer_structure.py` (db2
  level 2) exercise the transforms through their consumers.
- no fixture and no dedicated unit test: nothing under `tests/` imports
  `DecimatedWaveletTransform` or `UndecimatedWaveletTransform` directly, and no
  stored pre-refactor tensors exist; haar reconstruction was only confirmed by
  an ad hoc CPU check described above.
- Orthonormality of db2/db4 taps is by construction from the constants; no test
  verifies them.

## Variants and options

`wavelet` in `available_wavelets()`; `level`; decimated vs undecimated class.
Exact inverse exists only for haar/db1 decimated. For one level, stateless Haar
use `haar_dwt1d`.

## When to use and when not to use

Use decimated for multi-level pyramids on `[B, C, L]` (invertible with haar),
undecimated when each subband must align per timestep (no inverse). Do not use
where reconstruction with db2/db4 is required, where the length is shorter
than the undecimated receptive field, or when a single shared decomposition
instance is called on varying lengths before `reconstruct`.

## Related components

`haar_dwt1d` (stateless one-level Haar), `series_decomposition` (moving-average
trend/seasonal split), `freq_band_moe` and `frequency_band_sampler`
(frequency-domain band alternatives).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `DecimatedWaveletTransform(wavelet: str='haar', level: int=1)`
  Critically sampled multi-level orthogonal wavelet analysis/synthesis.
- `UndecimatedWaveletTransform(wavelet: str='db4', level: int=3)`
  A-trous stationary wavelet decomposition (no downsampling).
- `available_wavelets()`
  Return the supported wavelet names.

```python
from tsflab.models._components.wavelet import DecimatedWaveletTransform, UndecimatedWaveletTransform, available_wavelets
```

## Retrieval terms

`dwt`, `haar`, `multi-resolution`, `subband`, `undecimated`, `wavelet`

## Current model consumers (3)

`awemixer`, `dpwmixer`, `wdformer`
<!-- component-card:generated:end -->
