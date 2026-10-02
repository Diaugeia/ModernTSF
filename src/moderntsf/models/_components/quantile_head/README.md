---
name: "quantile_head"
kind: "component"
module: "moderntsf.models._components.quantile_head"
summary: "Non-crossing quantile head: median anchor from Linear(base) plus cumulative softplus gaps to ascending/descending quantile levels, input-conditioned."
category: "head"
input: "[batch, pred_len, channels, in_features]"
output: "[batch, pred_len, channels, Q] ascending along the last axis, ordered like quantile_levels"
origin: "ModernTSF probabilistic-forecasting rail (commit b9946b39, 'probabilistic forecasting', #20); monotone-gap construction is the repo's own design, no paper recorded"
origin_models: ["quantile_dlinear", "quantile_patchtst", "mqrnn", "tirex"]
tags: ["monotone", "non-crossing", "probabilistic", "quantile", "softplus", "cumulative"]
---

# quantile_head

## Purpose

Given per-step base features `base [B, L, C, F]`, `QuantileHead(levels, F)` emits
`Q = len(levels)` quantiles that cannot cross. With `m` the index of the level
closest to 0.5:

- `anchor = Linear(F -> 1)(base)` and `gaps = softplus(Linear(F -> Q-1)(base))` (each gap >= 0).
- `out[m] = anchor`; for `i > m`: `out[i] = anchor + sum_{j=m..i-1} gap_j`; for `i < m`: `out[i] = anchor - sum_{j=i..m-1} gap_j`.

So the output is non-decreasing in the level index by construction, and interval
width depends on the input.

## Origin and granularity

Introduced with the first probabilistic-forecasting batch (commit `b9946b39`, PR
#20, originally the file `_quantile_head.py` under the then-flat models directory) for TiRex-, QuantileDLinear-,
QuantilePatchTST-, and MQRNN-style models, then moved into `_components`
(`33ea2050`) and extended with level validation (`db5b2970`). The construction is
a repository design choice; no paper is recorded for the anchor-plus-gaps scheme,
and the models using it differ from their published quantile heads accordingly. The pinball
loss, quantile levels from `config.evaluation.quantile_levels`, and the
trunk producing `base` stay model-local.

## Interface

Public symbols: `QuantileHead`, `validate_quantile_levels`, `DEFAULT_QUANTILE_LEVELS`.

- `DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, ..., 0.9)`.
- `validate_quantile_levels(values)`: `None` returns the defaults as a list; otherwise returns the list after checking it is non-empty, each level strictly inside (0, 1), and strictly increasing; raises `ValueError` otherwise.
- `QuantileHead(quantile_levels: list[float], in_features: int = 1)`: levels are validated as above. `in_features` is the trailing width of `base` (use 1 for a scalar anchor per step, after `unsqueeze(-1)`). Attributes `q`, `median_idx`, `in_features`.
- `forward(base [B, L, C, in_features]) -> [B, L, C, Q]` (any leading shape also works because only the last axis is projected, but the output is documented for rank 4). `Q == 1` returns just the anchor. Raises the `Linear` shape error when the trailing width mismatches.
- Parameters and state-dict keys: `anchor_proj.weight/bias` (`[1, F]`), `offset_proj.weight/bias` (`[Q-1, F]`). The levels are a non-persistent buffer `_levels` and are not in the state dict, so the checkpoint does not record the levels.
- Even-length level lists with 0.5 absent pick the level closest to 0.5 (ties go to the lower index) as the anchor; the anchor is then not an exact median.

## Invariants and equivalence evidence

no fixture. `test_quantile_head_is_monotone_and_differentiable` in
`tests/test_repository_contracts.py` checks the `[2, 8, 3, 3]` output shape,
`output[..., 1:] >= output[..., :-1]`, gradient finiteness, the default median
level, and rejection of invalid level lists. `tests/test_probabilistic_forecasters.py`
checks that `mqrnn` equals `quantile_head(local_decoder(...))` and its
`[2, 3, 2, 9]` output shape.

## Variants and options

The level set is configurable (any strictly increasing set in (0, 1)); odd
`Q` with 0.5 included makes the anchor the exact median. Not covered: crossing
heads that fit each level independently, heteroscedastic Gaussian heads, or
learned level embeddings (see `gaussian_parameter_head`).

## When to use and when not to use

Use for models whose probabilistic output is a rank-4 quantile grid scored with a
pinball/CRPS-style loss and where non-crossing is required. Do not use when
independent per-level heads are wanted (it couples levels through the anchor), when
the distribution is parametric, or when levels must be stored with the checkpoint.

## Related components

`gaussian_parameter_head` (parametric probabilistic output), `dlinear` and
`patchtst` (backbones feeding the anchor in the quantile consumers),
`flatten_forecast_head` (point-forecast head).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `QuantileHead(quantile_levels: list[float], in_features: int=1)`
  Monotone quantile head producing a non-crossing (B, L, C, Q) grid.
- `validate_quantile_levels(values: list[float] | tuple[float, ...] | None)`
  Return a validated, strictly increasing quantile-level list.
- `DEFAULT_QUANTILE_LEVELS`
  Public module constant.

```python
from moderntsf.models._components.quantile_head import QuantileHead, validate_quantile_levels, DEFAULT_QUANTILE_LEVELS
```

## Retrieval terms

`monotone`, `non-crossing`, `probabilistic`, `quantile`

## Current model consumers (4)

`mqrnn`, `quantile_dlinear`, `quantile_patchtst`, `tirex`
<!-- component-card:generated:end -->
