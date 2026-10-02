---
name: "marks"
kind: "component"
module: "moderntsf.models._components.marks"
summary: "Adapters from raw six-column timestamp marks to normalized calendar features, spatiotemporal (value + covariate) tensors, and Time-Series-Library mark layouts."
category: "utility"
input: "values [batch, time, nodes]; marks raw [batch, time, 6] = [year, month, day, weekday, hour, minute] or node covariates [batch, time, nodes, features]"
output: "normalized_time_features [batch, time, 2]; to_spatiotemporal [batch, time, nodes, 1 + F]; future_time_features [batch, time, nodes, F]; adapt_tslib_marks [batch, time, k]"
origin: "ModernTSF's own input contract; the value-plus-normalized-calendar layout follows the BasicTS / LargeST tensor convention as stated in the module docstring"
origin_models: []
tags: ["calendar", "covariate", "spatiotemporal", "timestamp", "time-in-day", "day-in-week", "adapter", "time-series-library"]
---

# marks

## Purpose

Pure tensor helpers (no parameters) that turn the framework's decoder-style
4-tuple marks into what model families expect.

- Raw marks are `[year, month, day, weekday, hour, minute]` per step.
- `normalized_time_features`: `time_in_day = (hour * 60 + minute) / 1440` and `day_in_week = weekday / 7`, stacked on the last axis.
- `to_spatiotemporal(values, marks)`: `cat([values[..., None], feats], -1)` where `feats` is the 4-D `marks` as given, or the two normalized features broadcast over the `N` nodes.
- `to_calendar_spatiotemporal`: same, but 4-D covariates must have exactly 2 channels.
- `future_time_features(marks, n)`: only the `feats` part, `[B, T, N, F]`.
- `coerce_time_length(marks, length)`: keep the last `length` steps, or repeat the last step to pad.
- `adapt_tslib_marks`: convert raw marks to the Time-Series-Library layout.

## Origin and granularity

Added with the PoorOtterBob spatiotemporal/air-quality models (commit `1b0e2f7d`,
"Add six PoorOtterBob models") and the forecasting data-setting modes (`7b7c4704`),
then extended by later refactors (`ef3dd5d2` made the model/data contracts
explicit). The module docstring states the layout follows the BasicTS / LargeST
convention, restricted to two calendar features by user specification; no single
paper is recorded. It is cut at the data-contract boundary so that node-structured models,
air-quality models, and TSLib transformers read the same marks consistently.
Model-specific handling stays local: how embeddings consume the features, and
any model that needs other calendar features.

## Interface

Public symbols: `TIME_FEATURES`, `TSLIB_TIME_FEATURE_DIMS`,
`tslib_time_feature_dimension`, `adapt_tslib_marks`, `normalized_time_features`,
`to_spatiotemporal`, `to_calendar_spatiotemporal`, `future_time_features`,
`coerce_time_length`. All are stateless tensor ops on float tensors; none has parameters.

- `TIME_FEATURES = 2`: count of normalized calendar channels.
- `TSLIB_TIME_FEATURE_DIMS`: `{"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}`, the pinned TSLib `timeF` widths.
- `tslib_time_feature_dimension(freq)`: case-insensitive lookup; raises `ValueError` listing supported keys.
- `adapt_tslib_marks(marks, *, embed_type, freq)`: `None` returns `None`; `marks.ndim != 3` raises `ValueError`. For `embed_type != "timeF"`: width 6 returns `marks[..., 1:]` (drops year), width 5 passes through, otherwise `ValueError`. For `"timeF"`: width equal to the pinned width passes through; width 6 with `freq == "h"` is converted to `[hour/23 - 0.5, weekday/6 - 0.5, (day-1)/30 - 0.5, (day_of_year-1)/365 - 0.5]` (leap-year aware); anything else raises `ValueError` (only hourly can be reconstructed from raw marks).
- `normalized_time_features(marks [B, T, 6]) -> [B, T, 2]`: reads columns 3 (weekday), 4 (hour), 5 (minute); no shape validation, values are not clamped.
- `to_spatiotemporal(values [B, T, N], marks) -> [B, T, N, 1 + F]`: `marks` 4-D is used unchecked; `None` is replaced by zeros `[B, T, 6]` (so both features become 0); otherwise `marks` is read as `[B, T, 6]`.
- `to_calendar_spatiotemporal(values, marks)`: raises `ValueError` if `marks` is 4-D with last dim != 2 (embedding-index models would otherwise index out of range), else delegates.
- `future_time_features(marks, n) -> [B, T, N, F]`: 4-D returned as-is; else normalized features expanded to `n` nodes (expand, not copy). `marks=None` is not handled (raises).
- `coerce_time_length(marks, length)`: adjusts dim 1 only; works for raw or node-structured marks. Padding repeats the last step; an empty time axis is not handled.

## Invariants and equivalence evidence

no fixture. `test_shared_spatiotemporal_adapter_shape` in
`tests/test_repository_contracts.py` checks `[2, 12, 4, 3]` output and that channel 0
equals the values. `tests/test_component_extraction_graph.py` runs many graph
consumers on `to_spatiotemporal` output and uses `normalized_time_features` in a
decoder reference model. The TSLib adapter functions have no direct test and no
current model consumer in `src/`.

## Variants and options

- Raw 3-D marks (forecasting datasets) vs. 4-D node covariates (air-quality datasets): same function, branch on `marks.dim()`.
- `to_calendar_spatiotemporal` restricts to 2-channel calendar covariates for models that index embedding tables by time-in-day and day-in-week.
- Time-of-day scale assumes minute-resolution stamps; `weekday` is assumed to be 0 to 6.

## When to use and when not to use

Use to feed graph / spatiotemporal / air-quality models and to build future
covariate blocks from decoder marks. Do not use when a model needs year, month,
or day features (they are discarded here), when marks have a different column
layout, or when sampling resolution is finer than one minute. Do not rely on it to
validate marks; most functions assume the layout.

## Related components

`embed` (consumes TSLib-layout marks), `forecast_embedding` (embeds raw six-column
marks directly), `graph_utils` (graph supports used by the same consumers).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- Import the module and use its documented functions/classes.

```python
import moderntsf.models._components.marks
```

## Retrieval terms

`calendar`, `covariate`, `spatiotemporal`, `timestamp`

## Current model consumers (38)

`agcrn`, `aircade`, `airdualode`, `airformer`, `airphynet`, `astgcn`, `bigst`, `bist`, `cauair`, `d2stgnn`, `dcrnn`, `deepair`, `dfdgcn`, `dgcrn`, `extralonger`, `gagnn`, `gclstm`, `gts`, `gwnet`, `himnet`, `lstm`, `mage`, `megacrn`, `mtgnn`, `pcdcnet`, `pm25gnn`, `ragc`, `st_ssdl`, `staeformer`, `stdn`, `stgcn`, `stgode`, `stid`, `stnorm`, `stop`, `sttn`, `stwave`, `visifold`
<!-- component-card:generated:end -->
