---
name: "last_value_center"
kind: "component"
module: "moderntsf.models._components.last_value_center"
summary: "Subtract the detached last time step from a [B, L, C] history before a head, and add it back to the head output."
category: "normalization"
input: "x [batch, length, channels]; head_output [batch, horizon, channels]; level [batch, 1, channels]"
output: "center_on_last_value: (centered [batch, length, channels], level [batch, 1, channels]); restore_last_value: [batch, horizon, channels]"
origin: "Last-value (NLinear-style) normalization, Zeng et al., 'Are Transformers Effective for Time Series Forecasting?', AAAI 2023"
origin_models: ["nlinear", "segrnn", "crossgnn"]
tags: ["centering", "detach", "last-value", "level", "residual", "stateless"]
---

# last_value_center

## Purpose

Two stateless functions that remove and restore the most recent level of a
history window. With `level = x[:, -1:, :].detach()`:

- `center_on_last_value(x) -> (x - level, level)`;
- `restore_last_value(head_output, level) -> head_output + level`.

No scaling is applied. The detach means no gradient reaches the model through
the additive level term (the centered path still gets gradient through `x`).

## Origin and granularity

The scheme is NLinear's last-value subtraction. The helper pair was added in
commit `b1518394` ("last_value_center; migrate TimeMixer to
series_decomposition"); the commit message names the consumers NLinear,
SegRNN and CrossGNN (its anti-OOD branch), and states that CATS, HL and
PatchTST keep their own variants with documented divergences. It is cut at the
arithmetic only: where the call goes, the head, padding, and the choice to
bypass (CrossGNN passes `0.0` instead when `anti_ood=False`) stay local. The
state is explicit: the caller keeps `level`, so the pair is stateless,
unlike `revin`.

## Interface

`center_on_last_value(x)`

- `x`: floating tensor `[batch, length, channels]` (indexed as
  `x[:, -1:, :]`; ranks other than 3 are not validated). `length >= 1`.
- Returns `(centered, level)`: `centered` has the shape of `x`; `level` is
  `[batch, 1, channels]`, detached (`requires_grad=False`).

`restore_last_value(head_output, level)`

- `head_output`: `[batch, horizon, channels]`; `level` broadcastable to it
  (a Python float such as `0.0` also works). Returns the sum with the
  broadcast shape. No validation, no parameters, no state-dict keys, no errors
  raised by the component.

## Invariants and equivalence evidence

- `tests/fixtures/component_extraction_batch7.pt` holds pre-refactor outputs,
  state dicts, and input gradients for `nlinear`, `segrnn`, and `crossgnn` (with
  `anti_ood` true and false); `tests/test_component_extraction_batch7.py`
  requires identical state-dict keys and values, outputs, and gradients
  (atol 1e-6).
- `test_nlinear_restores_the_last_observation` in
  `tests/test_compact_local_implementations.py` shows a zero head returns the
  last observation repeated over the horizon.

## Variants and options

None. For mean/std normalization with reversible statistics use `revin`
(`subtract_last=True` gives a stateful last-value-centred variant that still
divides by the standard deviation). Not extracted: CATS, HL and PatchTST local
variants (documented in the extraction commit, not detailed in the repository).

## When to use and when not to use

Use for a cheap level-shift correction in front of a linear or recurrent head
whose output has the same channel count. Do not use when the scale also drifts
(use `revin`), when the output channel layout differs from the input, or when
gradient through the level is required.

## Related components

`revin` (stateful mean/std alternative), `channel_wise_linear` (NLinear head),
`series_decomposition`, `adain_style_norm`.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `center_on_last_value(x: torch.Tensor)`
  Subtract the detached final timestep from every step of ``x``.
- `restore_last_value(head_output: torch.Tensor, level: torch.Tensor)`
  Add the centering ``level`` back onto a head's forecast.

```python
from moderntsf.models._components.last_value_center import center_on_last_value, restore_last_value
```

## Retrieval terms

`centering`, `detach`, `last-value`, `level`, `residual`

## Current model consumers (3)

`crossgnn`, `nlinear`, `segrnn`
<!-- component-card:generated:end -->
