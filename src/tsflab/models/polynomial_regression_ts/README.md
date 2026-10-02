---
name: "PolynomialRegressionTS"
summary: "PolynomialRegressionTS expands each channel's lag window with integer powers from one through the configured degree, then applies a shared linear map to the forecast horizon."
paper: "https://doi.org/10.1002/9781118625590"
paper_title: "Applied Regression Analysis"
venue: "Wiley"
year: 1998
---
# PolynomialRegressionTS

PolynomialRegressionTS expands each channel's lag window with integer powers from one through the configured degree, then applies a shared linear map to the forecast horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1002/9781118625590); title: Applied Regression Analysis; venue/year: Wiley / 1998
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PolynomialRegressionTS.toml`](../../../../configs/models/PolynomialRegressionTS.toml).

## Differences

This is an independent implementation from the cited polynomial-regression
description; no external source implementation was inspected or copied. It uses
integer powers of each lag independently, without cross-lag or cross-channel
interaction monomials, and learns a direct multi-horizon map with gradient
descent.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `degree=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Polynomial regression is a classical statistical technique that enriches the feature space of a linear model by including nonlinear transformations of the input variables. In the time-series forecasting context, the historical window values are expanded with squared and square-root lag features before a linear predictor maps them to the output horizon. This polynomial feature augmentation allows the model to capture simple nonlinear trends without the overhead of a deep neural network. The TSFLab implementation trains this model end-to-end as a `torch.nn.Module`, enabling execution on GPU/MPS via the standard training loop and making it a useful nonlinear classical baseline alongside purely linear methods.

## In TSFLab
Default config: `configs/models/PolynomialRegressionTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent implementation from the cited polynomial-regression
description; no external source implementation was inspected or copied. It uses
integer powers of each lag independently, without cross-lag or cross-channel
interaction monomials, and learns a direct multi-horizon map with gradient
descent.

## Citation

```bibtex
@book{draper1998applied,
  author    = {Norman R. Draper and Harry Smith},
  title     = {Applied Regression Analysis},
  edition   = {3rd},
  publisher = {Wiley},
  address   = {New York},
  year      = {1998},
  doi       = {10.1002/9781118625590}
}
```
