---
name: "ARIMATS"
summary: "ARIMATS is a differentiable conditional ARIMA(p,1,q) recurrence with shared coefficients, historical one-step innovations, and zero expected future innovations."
paper: "https://search.worldcat.org/title/Time-series-analysis-forecasting-and-control/oclc/1422106714"
paper_title: "Time Series Analysis: Forecasting and Control"
venue: "Holden-Day 1970"
year: 1970
tagline: "Differentiable conditional ARIMA(p,1,q) recurrence with shared AR/MA coefficients fit by gradient descent."
tags: ["statistical", "autoregressive", "baseline", "lightweight", "channel-independent"]
composition: ["normalization=none", "decomposition=local:first-order-differencing", "temporal=local:conditional-arima-recurrence", "channel=local:channel-independent-shared-weights", "head=local:cumulative-sum-of-forecast-differences", "loss=loss:mse"]
---
# ARIMATS

## Key ideas

- Differences the input once (d=1) and runs the ARMA recurrence over the differences, with learnable `ar_coefficients`, `ma_coefficients` and `drift`.
- Historical one-step innovations are computed from the observed differences; future innovations are zero (conditional expectation).
- Forecasts are the last observed value plus the cumulative sum of predicted differences.
- Coefficients are shared across channels and fit by gradient descent, not likelihood; no order selection or seasonality.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://search.worldcat.org/title/Time-series-analysis-forecasting-and-control/oclc/1422106714); title: Time Series Analysis: Forecasting and Control; venue/year: Holden-Day 1970 / 1970
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ARIMATS.toml`](../../../../configs/models/ARIMATS.toml).

## Differences

This clean-room implementation fixes differencing order to one, estimates coefficients by gradient descent, shares them across channels, and uses conditional zero for unknown future innovations. It does not perform likelihood fitting, order selection, stationarity transforms, seasonal ARIMA, or confidence intervals. No third-party implementation was inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `ar_order=2`, `ma_order=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Time Series Analysis: Forecasting and Control
- **Venue**: Holden-Day (book); N/A (classical baseline)
- **Published**: 1970
- **Link**: https://search.worldcat.org/title/Time-series-analysis-forecasting-and-control/oclc/1422106714

## Abstract
ARIMA (Autoregressive Integrated Moving Average) is a classical statistical framework for modeling and forecasting univariate time series, introduced by Box and Jenkins (1970). An ARIMA(p,d,q) model combines autoregressive terms (AR), differencing to achieve stationarity (I), and moving-average terms (MA). The model captures linear temporal dependencies by regressing the current value on its own past values and on past forecast errors, after applying d rounds of differencing to remove trend non-stationarity. Model orders (p, d, q) are typically selected via the ACF/PACF plots and information criteria such as AIC/BIC. ARIMA remains a widely used baseline for short- and medium-term forecasting across economics, meteorology, and engineering.

## In TSFLab
Default config: `configs/models/ARIMATS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This clean-room implementation fixes differencing order to one, estimates coefficients by gradient descent, shares them across channels, and uses conditional zero for unknown future innovations. It does not perform likelihood fitting, order selection, stationarity transforms, seasonal ARIMA, or confidence intervals. No third-party implementation was inspected or copied.

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
