---
name: "gaussian_parameter_head"
kind: "component"
module: "tsflab.models._components.gaussian_parameter_head"
summary: "Two linear heads giving a Gaussian location and a strictly positive scale, softplus (or the explicit log(1 + exp) form) plus eps."
category: "head"
input: "[..., in_features]"
output: "(loc, scale), each [..., out_features]"
origin: "Gaussian likelihood parameter layer of DeepAR (Salinas et al., Int. J. Forecasting 2020); consumed by deepar and gaussian_mlp"
origin_models: ["deepar", "gaussian_mlp"]
tags: ["distribution", "gaussian", "location", "probabilistic", "scale", "softplus"]
---

# gaussian_parameter_head

## Purpose

`GaussianParameterHead(in_features, out_features)` returns an independent
Gaussian per output element: `loc = Linear_loc(x)` and
`scale = g(Linear_scale(x)) + eps`, with `g = softplus` or the algebraically equal
`log(1 + exp(.))`. The scale is strictly positive. It outputs parameters only; the
negative log-likelihood and sampling live in the loss and the model.

## Origin and granularity

The earliest history entry for this file is commit `4fac489c` ("make model
cards canonical"); its consumers are `deepar` and `gaussian_mlp`. The `log1pexp`
option exists to "preserve the official DeepAR expression for reference
comparison" (code comment); `deepar` uses the default `softplus` with
`eps=1e-6`. That `deepar` was the specific origin is inferred from this comment and
its consumer list; history does not record a different first author of the head.
Kept at the parameter-projection boundary: the recurrent trunk, the likelihood,
autoregressive sampling, and scale normalization of covariates remain in the models.

## Interface

`GaussianParameterHead(in_features: int, out_features: int, *, eps: float = 1e-6, scale_transform: "softplus" | "log1pexp" = "softplus")`

- `in_features`, `out_features` (int >= 1): last-axis widths of the input and of both outputs.
- `eps` (float > 0): added to the scale as a floor; `eps <= 0` raises `ValueError`.
- `scale_transform`: other values raise `ValueError`. `"log1pexp"` is `log(1 + exp(x))` computed naively and overflows to `inf` for large raw scale (above about 88 in float32); `"softplus"` is numerically stable.
- `forward(x [..., in_features]) -> (loc [..., out_features], scale [..., out_features])`; leading axes are arbitrary.
- State-dict keys: `loc_layer.weight`, `loc_layer.bias`, `scale_layer.weight`, `scale_layer.bias`. Stateless; float tensors on one device.

## Invariants and equivalence evidence

- Contract, invariant, gradient, and seeded numerical-regression tests: `tests/test_component_contracts_basic.py`, reference values in `tests/fixtures/components/gaussian_parameter_head_softplus.pt`, `tests/fixtures/components/gaussian_parameter_head_log1pexp.pt`.
no fixture. `test_gaussian_parameter_head_preserves_both_scale_formulas` in
`tests/test_repository_contracts.py` checks both transforms against their formulas
plus `eps`, strict positivity, and gradient finiteness.
`tests/test_probabilistic_forecasters.py` checks the `GaussianMLP` Gaussian
parameter equation and positive scale and the output shape through the model.

## Variants and options

`scale_transform` in `{"softplus", "log1pexp"}` and `eps`. Not covered: full
covariance, Student-t or negative-binomial heads, a fixed or shared scale.

## When to use and when not to use

Use when a model's output is an independent Gaussian per element and training uses
a Gaussian NLL. Do not use for correlated outputs, non-Gaussian likelihoods,
bounded targets, or when scale should be tied across horizons.

## Related components

`quantile_head` (alternative probabilistic output), `flatten_forecast_head` (point
forecast head with a different output type).

<!-- component-card:generated:start -->
## Public API

Implementation: [`__init__.py`](__init__.py)

- `GaussianParameterHead(in_features: int, out_features: int, *, eps: float=1e-06, scale_transform: Literal['softplus', 'log1pexp']='softplus')`
  Project features to independent Gaussian location and positive scale.

```python
from tsflab.models._components.gaussian_parameter_head import GaussianParameterHead
```

## Retrieval terms

`distribution`, `gaussian`, `location`, `probabilistic`, `scale`

## Current model consumers (2)

`deepar`, `gaussian_mlp`
<!-- component-card:generated:end -->
