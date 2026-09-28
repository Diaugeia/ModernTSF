---
name: "periodic_query_bank"
kind: "component"
module: "moderntsf.models._components.periodic_query_bank"
summary: "Learnable per-phase vector table gathered into phase-aligned windows."
---

# periodic_query_bank

## Purpose

Learnable per-phase vector table gathered into phase-aligned windows.

A learnable table of per-phase vectors gathered into phase-aligned windows.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `PeriodicQueryBank(period: int, channels: int)`
  One learnable vector per phase of a fixed period, indexed by offset.

```python
from moderntsf.models._components.periodic_query_bank import PeriodicQueryBank
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `periodic_query_bank` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `cycle`, `gather`, `period`, `phase`, `query`.

## Current model consumers

- [`tqnet`](../../tqnet/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
