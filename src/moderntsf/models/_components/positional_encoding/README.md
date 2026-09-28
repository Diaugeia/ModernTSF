---
name: "positional_encoding"
kind: "component"
module: "moderntsf.models._components.positional_encoding"
summary: "Patch-transformer positional encodings."
---

# positional_encoding

## Purpose

Patch-transformer positional encodings.

Position-table construction for patch-based sequence encoders.

Implementation: [`__init__.py`](__init__.py)

## Public API

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.positional_encoding
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `positional_encoding` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `encoding`, `patch`, `position`, `transformer`.

## Current model consumers

- No model currently declares this component directly.

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
