---
name: "marks"
kind: "component"
module: "moderntsf.models._components.marks"
summary: "Canonical temporal-mark and spatiotemporal input adapters."
---

# marks

## Purpose

Canonical temporal-mark and spatiotemporal input adapters.

Shared temporal-mark and spatiotemporal input-contract helpers.

Implementation: [`__init__.py`](__init__.py)

## Public API

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.marks
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `marks` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `calendar`, `covariate`, `spatiotemporal`, `timestamp`.

## Current model consumers

- [`agcrn`](../../agcrn/README.md)
- [`aircade`](../../aircade/README.md)
- [`airdualode`](../../airdualode/README.md)
- [`airformer`](../../airformer/README.md)
- [`airphynet`](../../airphynet/README.md)
- [`astgcn`](../../astgcn/README.md)
- [`bigst`](../../bigst/README.md)
- [`bist`](../../bist/README.md)
- [`cauair`](../../cauair/README.md)
- [`d2stgnn`](../../d2stgnn/README.md)
- [`dcrnn`](../../dcrnn/README.md)
- [`deepair`](../../deepair/README.md)
- [`dfdgcn`](../../dfdgcn/README.md)
- [`dgcrn`](../../dgcrn/README.md)
- [`extralonger`](../../extralonger/README.md)
- [`gagnn`](../../gagnn/README.md)
- [`gclstm`](../../gclstm/README.md)
- [`gts`](../../gts/README.md)
- [`gwnet`](../../gwnet/README.md)
- [`himnet`](../../himnet/README.md)
- [`lstm`](../../lstm/README.md)
- [`mage`](../../mage/README.md)
- [`megacrn`](../../megacrn/README.md)
- [`mtgnn`](../../mtgnn/README.md)
- [`pcdcnet`](../../pcdcnet/README.md)
- [`pm25gnn`](../../pm25gnn/README.md)
- [`ragc`](../../ragc/README.md)
- [`st_ssdl`](../../st_ssdl/README.md)
- [`staeformer`](../../staeformer/README.md)
- [`stdn`](../../stdn/README.md)
- [`stgcn`](../../stgcn/README.md)
- [`stgode`](../../stgode/README.md)
- [`stid`](../../stid/README.md)
- [`stnorm`](../../stnorm/README.md)
- [`stop`](../../stop/README.md)
- [`sttn`](../../sttn/README.md)
- [`stwave`](../../stwave/README.md)
- [`visifold`](../../visifold/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
