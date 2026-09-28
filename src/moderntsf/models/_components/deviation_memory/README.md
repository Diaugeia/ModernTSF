---
name: "deviation_memory"
kind: "component"
module: "moderntsf.models._components.deviation_memory"
summary: "Learnable prototype memory bank with attention retrieval and a deviation score between representations."
---

# deviation_memory

## Purpose

Learnable prototype memory bank with attention retrieval and a deviation score between representations.

Learnable prototype memory bank with attention retrieval and a paper-neutral deviation score between two representations.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `PrototypeMemory(query_dim: int, prototype_dim: int, num_prototypes: int)`
  Learnable prototype bank with soft attention retrieval and top-2 lookup.
- `PrototypeRetrieval()`
  Outputs of one :class:`PrototypeMemory` query.
- `deviation_score(current: torch.Tensor, reference: torch.Tensor)`
  Mean absolute deviation between two same-shaped representations.

```python
from moderntsf.models._components.deviation_memory import PrototypeMemory, PrototypeRetrieval, deviation_score
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `deviation_memory` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `contrastive`, `deviation`, `memory`, `prototype`, `retrieval`, `self-supervised`.

## Current model consumers

- [`st_ssdl`](../../st_ssdl/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
