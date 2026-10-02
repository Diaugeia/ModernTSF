---
name: "channel_alignment"
kind: "component"
module: "moderntsf.models._components.channel_alignment"
summary: "fit_channels slices or right-pads the last axis with zeros to exactly the requested width; no learnable parameters."
category: "utility"
input: "values [..., channels]; width int >= 1"
output: "[..., width]"
origin: "Repository utility extracted from graph recurrent forecasters; origin paper not applicable"
origin_models: ["dcrnn", "gclstm", "gts"]
tags: ["adapter", "channel", "feature", "padding", "shape", "slice", "stateless"]
---

# channel_alignment

## Purpose

`fit_channels(values, width)` makes the final axis exactly `width` wide
deterministically: if `values.shape[-1] >= width` it returns
`values[..., :width]` (keeping the leading channels), otherwise it appends
`width - C` zero channels on the right.

## Origin and granularity

Extracted in "extract reusable forecast contracts" (`db5b2970`) to replace
repeated model-local `_fit_channels` helpers; the repository contract test
`test_repeated_model_helpers_are_extracted` forbids reintroducing a local
`_fit_channels`. Current consumers are the spatiotemporal recurrent models
`dcrnn`, `gclstm` and `gts`, which pass `fit_channels(to_spatiotemporal(x_enc,
x_mark_enc), input_dim)` to align the (value + mark) features with a fixed cell
input width. The original per-model helpers' exact origin is not recorded in
history. What stays local: choosing `width`, building the marks, and the model
that consumes the aligned tensor.

## Interface

`fit_channels(values, width)`

- `values`: any tensor with at least one axis; only the last axis is touched.
- `width` (int >= 1): target size. Raises `ValueError("width must be positive")`
  for `width < 1`.
- Returns `[..., width]`. When `C >= width` this is a view slice (never the
  same object, even if `C == width`); when `C < width` it is a new tensor from
  `values.new_zeros` plus `torch.cat`, so dtype and device follow `values`.
- No parameters or state; no gradient for the padded zeros, gradient flows to the
  kept slice.
- Extra channels are silently dropped, which can discard information.

## Invariants and equivalence evidence

- `test_shared_channel_alignment_and_forecast_embedding_contracts` in
  `tests/test_repository_contracts.py` checks the slice case, the zero-padded
  case (leading channels preserved, padding is zeros), and the `width=0` error.
- no fixture: no pre-refactor tensor fixture; the three consumers are covered
  by the generic model contract tests.

## Variants and options

None. Left-padding, learned projections, or reflect padding are not provided.

## When to use and when not to use

Use to adapt an input whose feature count may be smaller or larger than a cell's
fixed input width, and where dropping trailing channels is acceptable. Do not
use when all channels matter and the width is smaller (use a learned projection),
or when padding should carry a mask.

## Related components

`marks` (`to_spatiotemporal` builds the tensor typically aligned),
`forecast_embedding`, `channel_wise_linear`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `fit_channels(values: torch.Tensor, width: int)`
  Slice or right-pad the final axis to exactly ``width`` channels.

```python
from moderntsf.models._components.channel_alignment import fit_channels
```

## Retrieval terms

`adapter`, `channel`, `feature`, `padding`, `shape`

## Current model consumers (3)

`dcrnn`, `gclstm`, `gts`
<!-- component-card:generated:end -->
