---
name: "soft_tree"
kind: "component"
module: "moderntsf.models._components.soft_tree"
summary: "Differentiable binary and level-wise-shared tree routing with leaf interpolation."
---

# soft_tree

## Purpose

Differentiable binary and level-wise-shared tree routing with leaf interpolation.

Paper-neutral differentiable binary-tree routing primitives.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `SoftDecisionTree(input_dim: int, output_dim: int, depth: int=3, temperature: float=1.0, *, split_mask: torch.Tensor | None=None, fixed_split_weight: torch.Tensor | None=None, fixed_threshold: torch.Tensor | None=None)`
  Interpolate leaf values with differentiable binary path probabilities.
- `SoftObliviousTree(input_dim: int, output_dim: int, depth: int=3, temperature: float=1.0)`
  Soft tree whose nodes at the same depth share one split decision.
- `binary_routes(depth: int)`
  Return heap node indices and right-branch indicators for every leaf.

```python
from moderntsf.models._components.soft_tree import SoftDecisionTree, SoftObliviousTree, binary_routes
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `soft_tree` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `decision`, `ensemble`, `leaf`, `oblivious`, `routing`, `soft`, `tree`.

## Current model consumers

- [`catboost_ts`](../../catboost_ts/README.md)
- [`decision_tree_ts`](../../decision_tree_ts/README.md)
- [`extra_trees_ts`](../../extra_trees_ts/README.md)
- [`gradient_boosting_ts`](../../gradient_boosting_ts/README.md)
- [`lightgbm_ts`](../../lightgbm_ts/README.md)
- [`random_forest_ts`](../../random_forest_ts/README.md)
- [`xgboost_ts`](../../xgboost_ts/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
