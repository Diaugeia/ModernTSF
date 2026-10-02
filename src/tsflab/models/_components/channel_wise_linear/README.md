---
name: "channel_wise_linear"
kind: "component"
module: "tsflab.models._components.channel_wise_linear"
summary: "Affine map from input_length to output_length over the last axis of [B, C, L], shared across channels or one independent nn.Linear per channel."
category: "head"
input: "[batch, channels, input_length]"
output: "[batch, channels, output_length]"
origin: "Temporal linear projection of the LTSF-Linear family (Zeng et al., 'Are Transformers Effective for Time Series Forecasting?', AAAI 2023): the shared / individual Linear layer of Linear, NLinear, DLinear"
origin_models: ["linear", "nlinear", "dlinear"]
tags: ["channel-wise", "forecast", "individual", "linear", "projection", "stateless"]
---

# channel_wise_linear

## Purpose

`ChannelWiseLinear(input_length, output_length, channels, individual)` maps the
history axis to the forecast axis. In shared mode one `nn.Linear(L, H)` is
applied to every channel; in individual mode channel `c` uses its own
`nn.Linear(L, H)`:

`y[b, c, :] = W_c x[b, c, :] + b_c` (with `W_c = W`, `b_c = b` for all `c` when shared).

Normalization, decomposition, and the transpose from `[B, L, C]` stay with the
caller.

## Origin and granularity

The shared-vs-individual `nn.Linear` pattern is the LTSF-Linear family's
temporal projection. It was extracted from the repeated local linear-layer
blocks of the LTSF-Linear models (commit `4fac489c`, "make model cards
canonical"). The cut is exactly the projection layer: normalization (`revin`,
`last_value_center`), decomposition (`series_decomposition`), layout permutes,
and any residual or mixing stay in the model. Direct consumers: `linear`,
`nlinear`, `rlinear`, `mtlinear`, `tsmixer`, `mtsmixer`, `rpmixer`, `distdf`,
`cosa`, `cyclenet`, `samformer`, `core`, plus the `composed` slot adapters;
`dlinear` uses it through the `dlinear` component. The generated block below
is the authoritative consumer list.

## Interface

`ChannelWiseLinear(input_length, output_length, channels, individual=False)`

- `input_length` (int >= 1), `output_length` (int >= 1), `channels` (int >= 1;
  only used for validation and, when `individual`, the number of layers).
  Constructor arguments are not validated; invalid values fail inside
  `nn.Linear`.
- Attributes `input_length`, `output_length`, `channels`, `individual`.
- Parameters and state-dict keys: shared mode `linear.weight` `[H, L]` and
  `linear.bias` `[H]`; individual mode `linears.{c}.weight` and
  `linears.{c}.bias` for `c` in `0..channels-1`. Weights use the default
  `nn.Linear` initialization (no `1/L` initialization is applied here;
  consumers like `cosa` overwrite `linear.weight` themselves).
- `forward(x)`: `x` is a floating tensor `[batch, channels, input_length]`.
  Raises `ValueError` if `x.ndim != 3` or `x.shape[1:]` differs from
  `(channels, input_length)`. Returns `[batch, channels, output_length]` in
  the dtype/device of the module.
- Stateless apart from the parameters. Individual mode loops over channels in
  Python, so it is slower for many channels.

## Invariants and equivalence evidence

- Contract, invariant, gradient, and seeded numerical-regression tests: `tests/test_component_contracts_basic.py`, reference values in `tests/fixtures/components/channel_wise_linear_shared.pt`, `tests/fixtures/components/channel_wise_linear_ind.pt`.
- `test_channel_wise_linear_matches_original_output_and_gradients` in
  `tests/test_repository_contracts.py`: shared and individual outputs and input
  gradients equal the direct `nn.Linear` formulation.
- `tests/test_component_extraction_mixer.py` keeps a frozen TSMixer reference
  that uses `ChannelWiseLinear` as its projection, so the state-dict keys of
  `tsmixer` (`projection.linear.*`) are pinned.
- no fixture: no pre-refactor tensor fixture for the LTSF-Linear models; their
  equations are checked by `tests/test_compact_local_implementations.py`.

## Variants and options

- `individual=False`: one map shared by all channels (parameters independent of
  `channels`).
- `individual=True`: `channels` independent maps.
- Hidden layers, nonlinearities, dropout, and flatten-style heads are not
  provided: see `flatten_forecast_head` for patch-flatten heads.

## When to use and when not to use

Use as the final temporal projection of a channel-independent linear or MLP
forecaster with a fixed history length and horizon. Do not use for variable
history length, when mixing across channels is needed (this layer never mixes
channels), or when the input is `[B, L, C]` without the caller transposing.

## Related components

`dlinear` (two projections over a decomposition), `series_decomposition`,
`last_value_center`, `revin`, `flatten_forecast_head` (flatten-then-linear
head over patches; use it when the input is patch tokens rather than a raw
`[B, C, L]` series), `mixer_block` (adds nonlinearity and channel mixing
around such projections), `weight_set_router` (interpolates between the shared and
individual modes by mixing a few weight sets per channel).
- `channel_alignment`: width adapter between channel counts, versus a per-channel temporal projection.
- `patchtst`: channel-independent backbone alternative.
- `fft_extrapolation_conv`: parameter-efficient frequency-domain history-to-horizon alternative.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `ChannelWiseLinear(input_length: int, output_length: int, channels: int, individual: bool=False)`
  Map ``(batch, channels, input_length)`` to a forecast length.

```python
from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
```

## Retrieval terms

`channel-wise`, `forecast`, `individual`, `linear`, `projection`

## Current model consumers (13)

`composed`, `core`, `cosa`, `cyclenet`, `distdf`, `linear`, `mtlinear`, `mtsmixer`, `nlinear`, `rlinear`, `rpmixer`, `samformer`, `tsmixer`
<!-- component-card:generated:end -->
