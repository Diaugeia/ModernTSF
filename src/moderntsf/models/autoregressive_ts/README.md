---
name: "AutoRegressiveTS"
summary: "AutoRegressiveTS is a classical autoregressive lag model for univariate and multivariate time-series forecasting. It directly maps the historical input window to the future prediction window using a learned linear projection over lagged observations, and is wrapped as a PyTorch `nn.Module` so that it integrates with the standard ModernTSF training loop and can run on CUDA/MPS devices."
paper: "https://search.worldcat.org/title/1422106714"
paper_title: "Time Series Analysis: Forecasting and Control"
venue: "Holden-Day"
year: 1970
---
# AutoRegressiveTS

AutoRegressiveTS is a classical autoregressive lag model for univariate and multivariate time-series forecasting. It directly maps the historical input window to the future prediction window using a learned linear projection over lagged observations, and is wrapped as a PyTorch `nn.Module` so that it integrates with the standard ModernTSF training loop and can run on CUDA/MPS devices.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://search.worldcat.org/title/1422106714); title: Time Series Analysis: Forecasting and Control; venue/year: Holden-Day / 1970
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AutoRegressiveTS.toml`](../../../../configs/models/AutoRegressiveTS.toml).

## Differences

This is an independent implementation from the cited AR description; no external
source implementation was inspected or copied. Unlike a recursively fitted
one-step AR model, it learns one direct map from the complete lag window to all
forecast horizons. The coefficients are shared across channels and do not form a
full vector autoregression.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Autoregressive (AR) models predict the next value (or block of values) in a time series as a linear combination of a fixed number of past observations, known as the lag order. The parameters are typically estimated by ordinary least squares or Yule–Walker equations. When extended to the vector setting (VAR), each variable is regressed on its own lags and the lags of all other variables. The AR/VAR family is one of the oldest and most studied approaches in time-series analysis, forming the basis for more complex models such as ARIMA and state-space methods. In ModernTSF the model is implemented as a differentiable linear layer that maps the full input window to the full prediction horizon in a single forward pass, enabling end-to-end gradient-based training.

## In ModernTSF
Default config: `configs/models/AutoRegressiveTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent implementation from the cited AR description; no external
source implementation was inspected or copied. Unlike a recursively fitted
one-step AR model, it learns one direct map from the complete lag window to all
forecast horizons. The coefficients are shared across channels and do not form a
full vector autoregression.

## Citation

```bibtex
@book{box1970time,
  author    = {George E. P. Box and Gwilym M. Jenkins},
  title     = {Time Series Analysis: Forecasting and Control},
  publisher = {Holden-Day},
  address   = {San Francisco},
  year      = {1970},
  url       = {https://archive.org/details/timeseriesanalys0000boxg}
}
```
