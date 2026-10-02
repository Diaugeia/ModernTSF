---
name: "flatten_forecast_head"
kind: "component"
module: "tsflab.models._components.flatten_forecast_head"
summary: "Flatten the last two axes of [batch, channels, d, patches] and project to the horizon with one shared or one per-channel Linear, plus dropout."
category: "head"
input: "[batch, n_vars, d_model, patches] (any [batch, n_vars, ..., ...] whose last two axes flatten to nf)"
output: "[batch, n_vars, target_window]"
origin: "PatchTST flatten head (Nie et al., ICLR 2023, A Time Series is Worth 64 Words); extracted from the PatchTST backbone component and four model-local copies"
origin_models: ["patchtst", "hdmixer", "moderntcn", "timexer", "umixer"]
tags: ["channel-wise", "flatten", "forecast", "head", "linear", "patch", "individual", "patchtst", "point-forecast", "linear-head", "shared-head"]
---

# flatten_forecast_head

## Purpose

For an input `x` of shape `[B, C, D, P]` the head computes, per channel,
`dropout(Linear(D * P -> H)(flatten(x[:, c])))` and returns `[B, C, H]`. With
`individual=False` one `Linear` and one dropout are shared across channels; with
`individual=True` channel `c` owns its own `Linear`, flatten, and dropout.
The last two axes are flattened in their existing order (`D` major, `P` minor).

## Origin and granularity

Extracted in commit `d38451c3` ("extract flatten forecast head") from the
`patchtst` backbone component (`_components/patchtst`) and from local copies in the
`hdmixer`, `moderntcn`, `timexer`, and `umixer` models, which all used the
PatchTST-style flatten-then-linear head (`origin_models` lists these as the historical
sources; the commit does not name the paper, and PatchTST, Nie et al., ICLR 2023, is
the standard source of that head). Those four models no longer import the head; the
current direct consumers are the generated list below, and `quantile_patchtst`
reaches it through the `patchtst` component. The `patchtst` model keeps its own head
code. Cut at the head only: the preceding encoder, the normalization, and any reshape
into `[B, C, D, P]` stay model-local.

## Interface

`FlattenForecastHead(individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0)`

- `individual` (bool): per-channel parameters when true.
- `n_vars` (int >= 1): channel count. Only used when `individual=True` (loop bound); the shared head ignores it.
- `nf` (int >= 1): `D * P`, the flattened feature width; must equal the product of the last two axes of the input.
- `target_window` (int >= 1): horizon `H`.
- `head_dropout` (float in [0, 1)): dropout on the output.
- `forward(x)`: shared mode accepts any `[..., D, P]` (leading axes preserved) and returns `[..., H]`; individual mode requires rank 4 `[B, n_vars, D, P]` and returns the stack `[B, n_vars, H]`. A feature-width mismatch raises the `Linear` shape `RuntimeError`; no explicit validation.
- State-dict keys: shared `linear.weight`, `linear.bias`; individual `linears.{i}.weight`, `linears.{i}.bias`. Flatten and dropout have no parameters. Stateless apart from dropout.

## Invariants and equivalence evidence

- `tests/test_component_contracts_basic.py`:
  `test_flatten_forecast_head_contract_and_reference` (shared and individual) checks the
  exact state-dict keys, the `[2, 3, 5]` output and dtype, gradients to the input and all
  parameters, leading axes preserved in shared mode (`[7, 2, 3, 5]`), the `RuntimeError`
  for a feature-width mismatch, and seeded values and input gradient against
  `tests/fixtures/components/flatten_forecast_head_shared.pt` and
  `tests/fixtures/components/flatten_forecast_head_ind.pt`;
  `test_flatten_forecast_head_dropout_eval_deterministic` checks that eval mode with
  `head_dropout=0.5` is deterministic.
- `tests/test_repository_contracts.py`
  (`test_flatten_forecast_head_shared_and_individual_contracts`) checks that shared
  output equals `linear(flatten(x))`, that individual output equals the stack of
  per-channel linears, and the `[2, 3, 7]` shapes.
- No frozen pre-extraction fixture exists; extraction equivalence for the original
  consumers is not recorded as a test.

## Variants and options

`individual=True` for channel-specific heads (parameters scale with `n_vars`);
`head_dropout` for output dropout. Heads that add a nonlinearity, mix channels, or
project in two stages are not covered and stay local.

## When to use and when not to use

Use after a patch or token encoder whose output is arranged `[B, C, D, P]` and
whose forecast is a linear function of all `D * P` features per channel. Do not use
when the horizon should be produced autoregressively, when patch count changes at
run time (`nf` is fixed), or when channel mixing is required in the head.

## Related components

`channel_wise_linear` (per-channel linear over the time axis of `[B, L, C]`; this head
instead flattens a feature-by-patch grid), `patchtst` (backbone component that composes
this head), `embed` (`PatchEmbedding` produces the tokens such heads read), `dlinear`
(alternative linear forecaster), `quantile_head` and `gaussian_parameter_head`
(probabilistic output heads; this one emits a point forecast).
- `fft_extrapolation_conv`: frequency-domain history-to-horizon alternative.

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `FlattenForecastHead(individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float=0.0)`
  Map ``(B, C, D, P)``-like inputs to ``(B, C, horizon)``.

```python
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
```

## Retrieval terms

`channel-wise`, `flatten`, `forecast`, `head`, `linear`, `patch`

## Current model consumers (12)

`composed`, `gateformer`, `lsinet`, `mambats`, `mou`, `patchtsmixer`, `penguin`, `samba`, `semixer`, `sensorformer`, `srsnet`, `timeexpert`
<!-- component-card:generated:end -->
