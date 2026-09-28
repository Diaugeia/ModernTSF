---
name: "harmonic_energy_gate"
kind: "component"
module: "moderntsf.models._components.harmonic_energy_gate"
summary: "Per-channel harmonic-to-total spectral energy ratio for dual-branch gating."
---

# harmonic_energy_gate

## Purpose

Per-channel harmonic-to-total spectral energy ratio for dual-branch gating.

Per-channel harmonic-to-total spectral energy ratio for dual-branch gating.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `HarmonicEnergyGate(num_harmonics: int=3, low_freq_guard: int=3)`
  Return each channel's harmonic-energy share of its total spectral energy.

```python
from moderntsf.models._components.harmonic_energy_gate import HarmonicEnergyGate
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `harmonic_energy_gate` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `energy`, `fusion`, `gate`, `harmonic`, `periodicity`, `spectral`, `weighting`.

## Current model consumers

- [`dualformer`](../../dualformer/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
