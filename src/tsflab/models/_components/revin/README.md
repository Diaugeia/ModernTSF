---
name: "revin"
kind: "component"
module: "tsflab.models._components.revin"
summary: "Reversible instance normalization: per-instance mean/std (or last-value) normalization with optional learnable affine, inverted on the forecast."
category: "normalization"
input: "[batch, time, channels] (also any rank >= 3 with channels last)"
output: "same shape as input"
origin: "RevIN, Kim et al., ICLR 2022 (Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift)"
origin_models: ["patchtst"]
tags: ["denormalization", "instance", "normalization", "reversible", "distribution-shift", "affine", "stateful"]
---

# revin

## Purpose

`RevIN(num_features)` removes the per-instance level and scale of a history
window before a model, then puts them back on the model output. For one instance
`x` with time axis `t` and per-channel statistics taken over all axes between
batch and channels:

- `norm`: `mu = mean_t(x)` (or `x[:, -1]` when `subtract_last=True`),
  `sigma = sqrt(var_t(x, unbiased=False) + eps)`, `y = (x - mu) / sigma`, then
  `y = y * gamma + beta` when `affine=True`.
- `denorm`: undo the affine, `z = (y - beta) / (gamma + eps^2)`, then
  `z * sigma + mu`, using the `mu`/`sigma` cached by the latest `norm` call.

The statistics are detached, so gradients do not flow through the normalizer
statistics. The module is a stateful pair: `denorm` is only valid after `norm`.

## Origin and granularity

The block is the standard RevIN layer used by many forecasters; it was
consolidated from near-identical model-local copies (commit `61451843`,
"consolidate reversible normalization", replaced local copies in crib, glocalib,
mgsfformer, mofo, pathformer, timealign, timefilter, timekan, and timemixer) and the
`patchtst` backbone. It is cut at the layer boundary: normalization statistics,
affine parameters, and inversion. Model-specific uses stay local: where `norm`
and `denorm` are called, forecast-length handling, other normalizations
(per-patch, non-stationary de-stationary attention factors), and last-value
variants that do not store state (use `last_value_center` for those).

## Interface

`RevIN(num_features, eps=1e-5, affine=True, subtract_last=False, enabled=True)`

- `num_features` (int >= 1): channel width; must equal `values.shape[-1]`.
- `eps` (float > 0): variance floor added inside the square root; it also
  appears as `eps^2` in the denominator of the affine inversion.
- `affine` (bool): adds learnable `affine_weight` (init 1) and `affine_bias`
  (init 0), each `[num_features]`. These are the only parameters and the only
  state-dict keys; `affine=False` registers none.
- `subtract_last` (bool): center on the last time step instead of the mean;
  the scale is still the standard deviation.
- `enabled` (bool): when false, `forward` returns its input unchanged (same
  object) for both modes and builds no statistics.

`forward(values, mode)` with `mode` in `{"norm", "denorm"}`:

- `values`: floating tensor `[batch, time, channels]` or higher rank with the
  channel axis last (`ndim >= 3`). Statistics reduce over every axis from 1 to
  `ndim - 2`. In `denorm` the time length may differ from the history (for
  example the forecast horizon) because the cached statistics broadcast over it.
- Returns a tensor of the same shape and dtype/device as `values`.
- Raises `ValueError` for `ndim < 3`, a channel-width mismatch, or an unknown
  mode; `RuntimeError` when `denorm` precedes any `norm`. All of these checks are
  skipped when `enabled=False`. Constructor: `ValueError` for `num_features < 1`
  or `eps <= 0`.
- Rank > 3 quirk: the mean and variance reduce over every middle axis, but
  `subtract_last=True` takes the last step along axis 1 only (shape
  `[B, 1, ..., C]` by `select(1, -1)`), so for rank > 3 the center is not the last
  value over all middle axes. Only rank 3 is the documented contract.
- Cached `_center` and `_scale` are plain attributes (not buffers): they are not
  in the state dict, follow whichever instance last called `norm`, and make an
  instance unsafe to share across concurrent forward passes.

## Invariants and equivalence evidence

- `test_revin_contract_and_reference` in `tests/test_component_contracts_basic.py`
  (parametrized over `affine`/`subtract_last` = (T,F), (F,F), (T,T)) checks the
  state-dict keys, zero mean and unit variance when neither option is set, a zero
  last step for `subtract_last`, the `denorm(norm(x))` round trip, a different
  horizon length at `denorm`, gradient to the input and affine parameters, and the
  rank-4 shape; `test_revin_errors_and_disabled` checks the `RuntimeError` and
  `ValueError` cases and the identity of the disabled module. Reference values are in
  `tests/fixtures/components/revin_10.pt` (affine, mean), `tests/fixtures/components/revin_00.pt`
  (no affine) and `tests/fixtures/components/revin_11.pt` (affine, last-value); the
  suffix is `<affine><subtract_last>`.
- Round trip: `denorm(norm(x)) == x` up to float error for the default,
  `subtract_last=True`, and disabled configurations; checked by
  `test_revin_round_trip` in `tests/test_repository_contracts.py`.
- Disabled mode returns the identical tensor object (same test).
- `tests/test_component_extraction_graph.py` constructs RevIN inside a
  graph-consumer reference model; no dedicated pre-refactor fixture exists for
  RevIN (consolidation was checked through the consumers' contract tests).
- State-dict keys are `affine_weight` and `affine_bias` only, so checkpoints of
  models that held a local copy keep loading.

## Variants and options

- `affine=False` for pure statistics normalization.
- `subtract_last=True` for the last-value-centred variant: the level is the
  last observed step, the scale is still the standard deviation.
- `enabled=False` to ablate normalization without changing the call sites.
- Materially different normalizers (series stationarization with learned
  de-stationary factors, per-patch normalization, AdaIN with external statistics)
  are separate: see `adain_style_norm` and `last_value_center`.

## When to use and when not to use

Use when a model needs standard per-window mean/std normalization of `[B, L, C]`
inputs and the inverse on `[B, H, C]` outputs with the same channel count. Do
not use when the output channel count differs from the input (statistics are
per-channel), when the denormalization must happen on a different instance than
the one that normalized, when statistics should carry gradients, or when a paper
specifies non-detached or patch-wise statistics.

## Related components

- `last_value_center`: stateless last-value centering, no scaling; the caller keeps
  the level instead of the module.
- `adain_style_norm`: rescales to externally supplied statistics.
- `series_decomposition`: often applied after normalization (trend/residual split).
- `patchtst`: backbone that composes RevIN.
- `freq_band_moe`: a block that instance-normalizes internally (non-affine, not
  invertible by the caller), so do not stack it with RevIN unintentionally.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `RevIN(num_features: int, eps: float=1e-05, affine: bool=True, subtract_last: bool=False, enabled: bool=True)`
  Normalize one sequence instance and later restore its original scale.

```python
from tsflab.models._components.revin import RevIN
```

## Retrieval terms

`denormalization`, `instance`, `normalization`, `reversible`

## Current model consumers (91)

`adamshyper`, `amd`, `amplifier`, `amrc`, `aurora`, `awemixer`, `catboost_ts`, `cmos`, `composed`, `cora`, `crosslinear`, `cyclenet`, `decision_tree_ts`, `distdf`, `dpwmixer`, `dsformer`, `dualformer`, `duet`, `dynamic_tmoe`, `extra_trees_ts`, `fets`, `film`, `fredf`, `fredformer`, `ftp`, `glocalib`, `gpht`, `gradient_boosting_ts`, `gru_forecaster_ts`, `gtr`, `hdmixer`, `hn_mvts`, `implicitforecaster`, `interpdn`, `lightgbm_ts`, `lsinet`, `lstm_forecaster_ts`, `mambats`, `mgsfformer`, `mlp_forecaster_ts`, `moderntcn`, `mofo`, `mou`, `mtlinear`, `mtsmixer`, `nhits`, `occamvts`, `olinear`, `paifilter`, `patchtsmixer`, `patchtst`, `pathformer`, `penguin`, `pgn`, `phaseformer`, `phat`, `pmdformer`, `pws`, `random_forest_ts`, `refocus`, `rlinear`, `rnn_forecaster_ts`, `samba`, `samformer`, `sdmixer`, `semixer`, `sempo`, `srsnet`, `svtime`, `swift`, `symtime`, `tcn_forecaster_ts`, `texfilter`, `timealign`, `timebridge`, `timecap`, `timeemb`, `timefilter`, `timekan`, `timemachine`, `timemixer`, `timemosaic`, `timeperceiver`, `timexer`, `tqnet`, `tsrag`, `umixer`, `wavenet`, `wpmixer`, `xgboost_ts`, `xpatch`
<!-- component-card:generated:end -->
