---
name: "node_visibility"
kind: "component"
module: "moderntsf.models._components.node_visibility"
summary: "Node-level masking and subgraph grouping for scalable node-set attention."
---

# node_visibility

## Purpose

Node-level masking and subgraph grouping for scalable node-set attention.

Node-visibility masking and subgraph grouping for scalable node-set attention.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `random_mask_tokens(x: torch.Tensor, mask_ratio: float, generator: torch.Generator | None=None)`
  Randomly keep a ``(1 - mask_ratio)`` fraction of tokens along dim 1.
- `shuffle_tokens(x: torch.Tensor, generator: torch.Generator | None=None)`
  Apply an independent random per-sample permutation along dim 1.
- `unshuffle_tokens(x: torch.Tensor, perm: torch.Tensor)`
  Invert :func:`shuffle_tokens` given the permutation it returned.
- `group_into_subgraphs(x: torch.Tensor, subgraph_size: int)`
  Partition tokens into zero-padded fixed-size subgraph groups.
- `ungroup_subgraphs(x: torch.Tensor, num_groups: int, orig_length: int, subgraph_size: int)`
  Invert :func:`group_into_subgraphs`, dropping any padding tokens.

```python
from moderntsf.models._components.node_visibility import random_mask_tokens, shuffle_tokens, unshuffle_tokens, group_into_subgraphs, ungroup_subgraphs
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `node_visibility` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `graph`, `grouping`, `masking`, `node`, `sampling`, `subgraph`, `visibility`.

## Current model consumers

- [`visifold`](../../visifold/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
