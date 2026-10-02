---
name: "spectral_descriptor"
kind: "component"
module: "moderntsf.models._components.spectral_descriptor"
summary: "Per-window spectral entropy and low/mid/high band-energy ratios of the channel-averaged power spectrum."
---

# spectral_descriptor

## Purpose

Per-window spectral entropy and low/mid/high band-energy ratios of the channel-averaged power spectrum.

Window-level spectral descriptor: spectral entropy and three band-energy ratios.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `SpectralDescriptor(eps: float=1e-08)`
  Map ``[batch, length, channels]`` to ``[batch, 4]`` spectral statistics.

```python
from moderntsf.models._components.spectral_descriptor import SpectralDescriptor
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `spectral_descriptor` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `band`, `descriptor`, `energy`, `entropy`, `fft`, `power`, `ratio`, `spectral`, `spectrum`.

## Current model consumers

- [`core`](../../core/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
