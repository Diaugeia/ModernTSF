---
name: "freq_band_moe"
kind: "component"
module: "moderntsf.models._components.freq_band_moe"
summary: "Learned frequency-band decomposition with input-gated mixture-of-experts recombination."
---

# freq_band_moe

## Purpose

Learned frequency-band decomposition with input-gated mixture-of-experts recombination.

Frequency-band decomposition mixture of experts.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `FrequencyBandMixtureOfExperts(expert_num: int, seq_len: int)`
  Decompose a series into learned frequency bands and gate their mixture.

```python
from moderntsf.models._components.freq_band_moe import FrequencyBandMixtureOfExperts
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `freq_band_moe` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `band`, `decomposition`, `experts`, `frequency`, `gating`, `mixture`, `rfft`.

## Current model consumers

- [`freqmoe`](../../freqmoe/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
