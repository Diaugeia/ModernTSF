---
name: "energy_frequency_pooling"
kind: "component"
module: "moderntsf.models._components.energy_frequency_pooling"
summary: "Energy-weighted stochastic pooling of a complex spectrum across a token axis."
---

# energy_frequency_pooling

## Purpose

Energy-weighted stochastic pooling of a complex spectrum across a token axis.

Energy-weighted stochastic pooling across a token axis in the frequency domain.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `EnergyBasedFrequencyPooling()`
  Softmax-energy-weighted pooling of a complex spectrum across tokens.

```python
from moderntsf.models._components.energy_frequency_pooling import EnergyBasedFrequencyPooling
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `energy_frequency_pooling` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `energy`, `frequency`, `key-frequency`, `pooling`, `softmax`, `stochastic`.

## Current model consumers

- [`refocus`](../../refocus/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
