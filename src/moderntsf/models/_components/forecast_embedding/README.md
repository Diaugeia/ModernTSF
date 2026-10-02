---
name: "forecast_embedding"
kind: "component"
module: "moderntsf.models._components.forecast_embedding"
summary: "Linear value projection plus a bias-free linear projection of six raw calendar columns scaled by fixed maxima and shifted by 0.5, then dropout."
category: "embedding"
input: "values [batch, time, channels]; marks [batch, time, 6]"
output: "[batch, time, d_model]"
origin: "Embedding of the Autoformer/FEDformer decomposition models, written clean-room against the repository's six-column raw-timestamp contract (not a verbatim paper layer)"
origin_models: ["autoformer", "fedformer", "dualformer"]
tags: ["calendar", "covariate", "embedding", "forecast", "value", "raw-marks", "linear"]
---

# forecast_embedding

## Purpose

`ForecastEmbedding(channels, d_model, dropout)` returns
`dropout(Linear(channels -> d_model)(values) + RawCalendarEmbedding(marks))` with

`RawCalendarEmbedding(marks) = Linear(6 -> d_model, bias=False)(marks / [2100, 12, 31, 6, 23, 59] - 0.5)`

so each of the six raw columns `[year, month, day, weekday, hour, minute]` is
scaled by a fixed maximum into roughly `[-0.5, 0.5]` before projection. There is
no position term and no convolutional token embedding.

## Origin and granularity

Extracted in commit `db5b2970` ("extract reusable forecast contracts") from an
identical model-local `CalendarEmbedding` + `ForecastEmbedding` pair in
`autoformer`; `fedformer` and later `dualformer` (commit `dd63af6c`) consume the
same class. The original docstring says it embeds the six raw columns "without
source helpers"; which reference-implementation embedding it stands in for is not
recorded in history. Whether the exact constants (2100, 12, 31, 6, 23, 59) match any paper
is not recorded; they are the column ranges of the repository's timestamp. Cut
at the embedding boundary; decomposition, attention, and heads stay local.

## Interface

Public symbols: `RawCalendarEmbedding`, `ForecastEmbedding`.

- `RawCalendarEmbedding(d_model: int)`: parameter `projection.weight` `[d_model, 6]`. `forward(marks [B, T, 6]) -> [B, T, d_model]`; raises `ValueError` when `marks.ndim != 3` or the last dim is not 6. The scale vector is created on `marks`' dtype/device each call; no buffers.
- `ForecastEmbedding(channels: int, d_model: int, dropout: float)`: submodules `value` (`Linear(channels, d_model)` with bias), `calendar`, `dropout`. `forward(values [B, T, channels], marks [B, T, 6]) -> [B, T, d_model]`; raises `ValueError` for `values.ndim != 3` or when `marks` and `values` differ in batch or time. Marks are required (no `None` path). Float tensors on one device; `channels` must equal `values.shape[-1]`.
- State-dict keys: `value.weight`, `value.bias`, `calendar.projection.weight`. Stateless apart from dropout.

## Invariants and equivalence evidence

no fixture. `test_shared_channel_alignment_and_forecast_embedding_contracts` in
`tests/test_repository_contracts.py` checks the `[2, 5, 8]` output shape and the
`ValueError` on a time-axis mismatch. Behavioural identity with the pre-extraction
autoformer class is by identical source (see the diff of `db5b2970`), not by a
frozen fixture.

## Variants and options

Only `dropout` is configurable. A zero-dropout instance (`0.0`) is deterministic.
The calendar term cannot be disabled; supply zero marks (output of the calendar
term is then `projection(-0.5 * ones)`, a constant bias-like vector) if no covariates
exist.

## When to use and when not to use

Use for models consuming the repository's raw six-column marks at both encoder
and decoder (`x_mark_enc`, `x_mark_dec`) with a plain value projection. Do not use
when positions or circular-conv token embedding are required (`embed`'s
`DataEmbedding`), when marks have another width, or when marks may be `None`.

## Related components

`embed` (Time-Series-Library style embeddings), `marks` (mark adapters),
`series_decomposition` (decomposition used by the consuming models).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `RawCalendarEmbedding(d_model: int)`
  Project six raw calendar columns after fixed-range normalization.
- `ForecastEmbedding(channels: int, d_model: int, dropout: float)`
  Add projected values and normalized raw-calendar covariates.

```python
from moderntsf.models._components.forecast_embedding import RawCalendarEmbedding, ForecastEmbedding
```

## Retrieval terms

`calendar`, `covariate`, `embedding`, `forecast`, `value`

## Current model consumers (3)

`autoformer`, `dualformer`, `fedformer`
<!-- component-card:generated:end -->
