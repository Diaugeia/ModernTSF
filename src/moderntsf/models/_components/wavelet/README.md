---
name: "wavelet"
kind: "component"
module: "moderntsf.models._components.wavelet"
summary: "Fixed-filter decimated and a-trous (undecimated) discrete wavelet transforms for BCL tensors."
---

# wavelet

## Purpose

Fixed-filter decimated and a-trous (undecimated) discrete wavelet transforms for BCL tensors.

Fixed-filter discrete wavelet analysis and synthesis for BCL tensors.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `DecimatedWaveletTransform(wavelet: str='haar', level: int=1)`
  Critically sampled multi-level orthogonal wavelet analysis/synthesis.
- `UndecimatedWaveletTransform(wavelet: str='db4', level: int=3)`
  A-trous stationary wavelet decomposition (no downsampling).
- `available_wavelets()`
  Return the supported wavelet names.

```python
from moderntsf.models._components.wavelet import DecimatedWaveletTransform, UndecimatedWaveletTransform, available_wavelets
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `wavelet` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `dwt`, `haar`, `multi-resolution`, `subband`, `undecimated`, `wavelet`.

## Current model consumers

- [`awemixer`](../../awemixer/README.md)
- [`dpwmixer`](../../dpwmixer/README.md)
- [`wdformer`](../../wdformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
