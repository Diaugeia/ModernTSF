---
name: "dlinear"
kind: "component"
module: "tsflab.models._components.dlinear"
summary: "DLinear backbone: split [B, L, C] with edge-padded moving average into residual and trend, project each with its own ChannelWiseLinear, and sum to [B, H, C]."
category: "backbone"
input: "[batch, seq_len, c_in]"
output: "[batch, pred_len, c_in]"
origin: "DLinear, Zeng et al., 'Are Transformers Effective for Time Series Forecasting?', AAAI 2023 (LTSF-Linear); decomposition scheme from Autoformer, NeurIPS 2021"
origin_models: ["dlinear"]
tags: ["decomposition", "linear", "moving-average", "seasonal", "trend", "backbone", "stateless"]
---

# dlinear

## Purpose

`DLinearBackbone(c_in, seq_len, pred_len, kernel_size, individual)` forecasts
by decomposing the history, projecting each part linearly over time, and
adding:

`r, t = SeriesDecomposition(kernel_size)(x)` (`t` is the edge-padded moving
average, `r = x - t`), then
`y = ChannelWiseLinear_seasonal(r^T) + ChannelWiseLinear_trend(t^T)`, transposed
back to `[B, H, C]`.

## Origin and granularity

The scheme is DLinear from the LTSF-Linear paper; the moving-average decomposition
it uses is the Autoformer series decomposition. The backbone lived in the
original commit and was reduced when `series_decomposition` and
`channel_wise_linear` were extracted (commits `15bc27e6`, `4fac489c`), so this
component is now only the composition of the two with two projections. It is
shared because the same backbone feeds `dlinear`, `quantile_dlinear`
(followed by a quantile head), and `latenttsf` (applied on latent channels
`c_in = d_model`). Left local: input shape validation, normalization, any head
after the backbone, and the latent autoencoder of `latenttsf`.

## Interface

`DLinearBackbone(c_in, seq_len, pred_len, kernel_size=25, individual=False)`

- `c_in` (int >= 1): channels; `seq_len` (int >= 1): history length;
  `pred_len` (int >= 1): horizon.
- `kernel_size` (odd int >= 1, default 25): moving-average window; even or
  non-positive values raise `ValueError` from `EdgePaddedMovingAverage`. Larger
  than `seq_len` is accepted (edge padding repeats the end values).
- `individual` (bool): per-channel (`True`) or shared (`False`) projections.
- Attributes: `decomposition`, `seasonal_projection`, `trend_projection`,
  `seq_len`, `pred_len`, `channels`, `individual`.
- State-dict keys: `seasonal_projection.linear.{weight,bias}` and
  `trend_projection.linear.{weight,bias}` (shared), or
  `seasonal_projection.linears.{c}.*` / `trend_projection.linears.{c}.*`
  (individual). The decomposition has no parameters.
- `forward(x)`: `x` `[batch, seq_len, c_in]` -> `[batch, pred_len, c_in]`. The
  backbone does not validate `x`'s shape itself; `ChannelWiseLinear` raises
  `ValueError` if channels or length mismatch.
- No special weight initialization: the projections use the default `nn.Linear`
  initialization (no `1/seq_len` constant init is applied).

## Invariants and equivalence evidence

- Contract, invariant, gradient, and seeded numerical-regression tests: `tests/test_component_contracts_basic.py`, reference values in `tests/fixtures/components/dlinear_shared.pt`, `tests/fixtures/components/dlinear_ind.pt`.
- `test_dlinear_is_seasonal_plus_trend_forecasting` in
  `tests/test_compact_local_implementations.py` checks that the model output
  equals seasonal projection plus trend projection of the decomposition.
- `tests/test_probabilistic_attention_forecasters.py` exercises
  `quantile_dlinear` through this backbone.
- no fixture: no pre-refactor tensor fixture for the backbone itself; its parts
  are covered by `tests/test_repository_contracts.py` (decomposition,
  ChannelWiseLinear).

## Variants and options

- `individual` per-channel vs shared linear layers.
- `kernel_size` controls the trend smoothness (25 is the library default).
- For last-value centering instead of decomposition use `last_value_center`
  with `channel_wise_linear` (NLinear); for instance normalization use `revin`.

## When to use and when not to use

Use as a strong, cheap channel-independent baseline or as a base forecaster
inside a larger model. Do not use when covariates or time marks must enter, when
cross-channel interaction is required, or for variable-length inputs.

## Related components

`series_decomposition`, `channel_wise_linear`, `last_value_center`, `revin`,
`quantile_head` (used with it in `quantile_dlinear`).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `DLinearBackbone(c_in: int, seq_len: int, pred_len: int, kernel_size: int=25, individual: bool=False)`
  DLinear sequence-to-sequence forecasting backbone.

```python
from tsflab.models._components.dlinear import DLinearBackbone
```

## Retrieval terms

`decomposition`, `linear`, `moving-average`, `seasonal`, `trend`

## Current model consumers (4)

`dlinear`, `latenttsf`, `mtlinear`, `quantile_dlinear`
<!-- component-card:generated:end -->
