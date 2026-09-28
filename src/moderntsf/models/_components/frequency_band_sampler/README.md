---
name: "frequency_band_sampler"
kind: "component"
module: "moderntsf.models._components.frequency_band_sampler"
summary: "Depth-indexed contiguous frequency-band selection over an FFT axis."
---

# frequency_band_sampler

## Purpose

Depth-indexed contiguous frequency-band selection over an FFT axis.

Depth-indexed contiguous frequency-band selection over an FFT axis.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `HierarchicalFrequencySampler(num_layers: int, alpha: float=1.0)`
  Assign a depth-indexed contiguous frequency band out of ``num_layers``.

```python
from moderntsf.models._components.frequency_band_sampler import HierarchicalFrequencySampler
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `frequency_band_sampler` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `band`, `depth`, `fft`, `frequency`, `hierarchical`, `sampling`, `spectral`.

## Current model consumers

- [`dualformer`](../../dualformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
