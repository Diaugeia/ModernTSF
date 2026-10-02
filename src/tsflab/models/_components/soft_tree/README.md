---
name: "soft_tree"
kind: "component"
module: "tsflab.models._components.soft_tree"
summary: "Differentiable depth-d binary trees: sigmoid split probabilities multiplied along each path and used to interpolate learned leaf vectors; plain and level-shared (oblivious) forms."
category: "routing"
input: "[batch, input_dim] (rank-2 only)"
output: "[batch, output_dim]; leaf_probabilities gives [batch, 2**depth]; binary_routes gives (nodes [2**depth, depth] long, right [2**depth, depth] bool)"
origin: "origin not recorded in history; generic sigmoid-routed soft decision tree written for the repository tree baselines (decision tree, forest, extra trees, boosting)"
origin_models: ["decision_tree_ts", "random_forest_ts", "extra_trees_ts", "gradient_boosting_ts", "catboost_ts", "lightgbm_ts", "xgboost_ts"]
tags: ["decision", "ensemble", "leaf", "oblivious", "routing", "soft", "tree"]
---

# soft_tree

## Purpose

A depth-`D` tree has `2^D - 1` internal nodes (heap order, root 0, children
`2n+1`/`2n+2`) and `2^D` leaves. For `x: [B, F]`, node `n` sends `x` right with
probability `r_n = sigmoid((x @ (W_n * mask_n) - theta_n) / T)`. A leaf's
probability is the product over its path of `r_n` (right branch) or `1 - r_n`
(left); the output is `leaf_probs @ leaf_value` (`leaf_value: [2^D, output_dim]`).
Leaf probabilities sum to 1 per row. `SoftObliviousTree` uses one split per depth
level (`split_weight: [D, F]`, `threshold: [D]`) shared by every node of that level.

## Origin and granularity

Introduced as the shared primitive of the tree-ensemble baselines in the rewrite
commit `dc15d37d` ("replace tree and boosting adapters"), when the local
`soft_tree.py` held the algebra and every consumer kept only its ensemble
logic. No paper for the soft-tree algebra is recorded in history. Model-local: ensemble
construction (averaging for forest/extra-trees, stagewise residual stacking for
the boosting models, CatBoost's forecast-context linear maps), random seeds and
feature subsets, RevIN, and any training interpretation. These are differentiable
approximations, not the original greedy CART/GBDT algorithms.

## Interface

`SoftDecisionTree(input_dim, output_dim, depth=3, temperature=1.0, *, split_mask=None,
fixed_split_weight=None, fixed_threshold=None)`: all dims and `depth` >= 1,
`temperature > 0` (else `ValueError`). `split_mask` `[2^D-1, F]` (default all ones)
must keep at least one feature per node. `fixed_split_weight` `[2^D-1, F]` and
`fixed_threshold` `[2^D-1]` must be given together; then they are buffers (frozen)
and only `leaf_value` trains; otherwise `split_weight` (normal, std 1/sqrt(F)) and
`threshold` (zeros) are parameters. State-dict keys: `split_weight`, `threshold`,
`leaf_value` (`normal`, std 0.02), buffers `split_mask`, `route_nodes`,
`route_right` (the last two are structural and are saved too; with fixed splits
`split_weight`/`threshold` are buffers rather than parameters).
`SoftObliviousTree(input_dim, output_dim, depth=3, temperature=1.0)`: keys
`split_weight` `[D, F]`, `threshold` `[D]`, `leaf_value`, buffer `route_right`.
Both: `leaf_probabilities(x)` and `forward(x)` need `x` of shape `[batch, input_dim]`
(otherwise `ValueError`); leaves are ordered by path bits, most significant bit
first (leaf 0 is all-left). `binary_routes(depth)` returns heap node indices and
right-branch flags for every leaf (`ValueError` for `depth < 1`). Mask is applied
as `split_weight * split_mask` at every forward. No state across calls. Cost grows as `2^D`; keep depth small.

## Invariants and equivalence evidence

- `tests/test_tree_baselines.py`: `test_depth_one_tree_is_exact_sigmoid_leaf_interpolation`
  checks the depth-1 closed form with temperature 2 and that leaf probabilities
  sum to one; `test_oblivious_tree_shares_one_decision_per_depth` checks the
  level-shared tree; `test_named_models_keep_distinct_compositions` and
  `test_complete_runtime_contract` run all seven consumers.
- `tests/test_component_contracts_signal.py` pins the interface (shapes, dtype, state-dict keys, invariants, gradient flow, error cases) and a seeded numerical regression against `tests/fixtures/components/soft_tree.pt`.

## Variants and options

`split_mask` for fixed per-node feature subsets (random forest), fixed split
weights and thresholds for randomized frozen geometry (extra trees),
`SoftObliviousTree` for symmetric trees (CatBoost style), `temperature` for
routing sharpness (small is closer to hard splits). No hard routing, no
multi-output-per-leaf variants beyond a vector `leaf_value`.

## When to use and when not to use

Use for a differentiable tree/ensemble head on flat `[B, F]` features (for
example a flattened lag window). Do not use for rank-3 sequences without
flattening, large depth (exponential leaves), or when a faithful tree algorithm
(greedy splits, exact boosting) is needed.

## Related components

`topk_expert_router` (sparse mixture routing), `sparse_connection_router`, `revin`
(consumers normalize with it).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `SoftDecisionTree(input_dim: int, output_dim: int, depth: int=3, temperature: float=1.0, *, split_mask: torch.Tensor | None=None, fixed_split_weight: torch.Tensor | None=None, fixed_threshold: torch.Tensor | None=None)`
  Interpolate leaf values with differentiable binary path probabilities.
- `SoftObliviousTree(input_dim: int, output_dim: int, depth: int=3, temperature: float=1.0)`
  Soft tree whose nodes at the same depth share one split decision.
- `binary_routes(depth: int)`
  Return heap node indices and right-branch indicators for every leaf.

```python
from tsflab.models._components.soft_tree import SoftDecisionTree, SoftObliviousTree, binary_routes
```

## Retrieval terms

`decision`, `ensemble`, `leaf`, `oblivious`, `routing`, `soft`, `tree`

## Current model consumers (7)

`catboost_ts`, `decision_tree_ts`, `extra_trees_ts`, `gradient_boosting_ts`, `lightgbm_ts`, `random_forest_ts`, `xgboost_ts`
<!-- component-card:generated:end -->
