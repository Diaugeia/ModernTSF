---
name: "flatten_forecast_head"
kind: "component"
module: "tsflab.models._components.flatten_forecast_head"
summary: "Flatten the last two axes of [batch, channels, d, patches] and project to the horizon with one shared or one per-channel Linear, plus dropout."
category: "head"
input: "[batch, n_vars, d_model, patches] (any [batch, n_vars, ..., ...] whose last two axes flatten to nf)"
output: "[batch, n_vars, target_window]"
origin: "PatchTST flatten head (Nie et al., ICLR 2023, A Time Series is Worth 64 Words); extracted from the patchtst backbone and four model-local copies"
origin_models: ["patchtst", "hdmixer", "moderntcn", "timexer", "umixer"]
tags: ["channel-wise", "flatten", "forecast", "head", "linear", "patch", "individual"]
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
`patchtst` backbone and local copies in `hdmixer`, `moderntcn`, `timexer`, and
`umixer`, which all used the PatchTST-style flatten-then-linear head. The commit
does not name the paper; PatchTST (Nie et al., ICLR 2023) is the standard source of
that head, and `patchtst` is the first listed origin. Now also used by `gateformer`,
`srsnet`, `semixer`, `sensorformer`, `lsinet`, `timeexpert`. Cut at the head
only: the preceding encoder, the normalization, and any reshape into
`[B, C, D, P]` stay model-local.

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

- Contract, invariant, gradient, and seeded numerical-regression tests: `tests/test_component_contracts_basic.py`, reference values in `tests/fixtures/components/flatten_forecast_head_shared.pt`, `tests/fixtures/components/flatten_forecast_head_ind.pt`.
no fixture. `test_flatten_forecast_head_shared_and_individual_contracts` in
`tests/test_repository_contracts.py` checks that shared output equals
`linear(flatten(x))`, that individual output equals the stack of per-channel linears, the
`[2, 3, 7]` shapes, and gradient finiteness. Extraction equivalence for the
consumers was verified at commit time through their contract tests; no frozen
pre-extraction fixture exists.

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

`channel_wise_linear` (per-channel linear over time), `patchtst` (backbone that
composes this head), `embed` (`PatchEmbedding` produces the tokens such heads read),
`dlinear` (alternative linear forecaster).

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
