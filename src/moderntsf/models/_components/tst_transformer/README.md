---
name: "tst_transformer"
kind: "component"
module: "moderntsf.models._components.tst_transformer"
summary: "Time-series Transformer encoder blocks."
---

# tst_transformer

## Purpose

Time-series Transformer encoder blocks.

Compact Transformer encoder used by independent patch forecasters.

Implementation: [`__init__.py`](__init__.py)

## Public API

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.tst_transformer
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `tst_transformer` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `attention`, `encoder`, `time-series`, `transformer`.

## Current model consumers

- No model currently declares this component directly.

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
