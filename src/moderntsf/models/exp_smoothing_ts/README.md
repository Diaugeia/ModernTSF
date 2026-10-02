---
name: "ExpSmoothingTS"
summary: "ExpSmoothingTS is a differentiable simple-exponential-smoothing baseline. It learns one smoothing coefficient per channel, recursively updates the level, and repeats the final level across the forecast horizon."
paper: "https://doi.org/10.1016/j.ijforecast.2003.09.015"
paper_title: "Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages"
venue: "International Journal of Forecasting"
year: 2004
---
# ExpSmoothingTS

ExpSmoothingTS is a differentiable simple-exponential-smoothing baseline. It learns one smoothing coefficient per channel, recursively updates the level, and repeats the final level across the forecast horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1016/j.ijforecast.2003.09.015); title: Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages; venue/year: International Journal of Forecasting / 2004
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ExpSmoothingTS.toml`](../../../../configs/models/ExpSmoothingTS.toml).

## Differences

This is an independent implementation of the simple level-only exponential
smoothing recurrence; no external source implementation was inspected or
copied. It omits Holt trend and seasonal states, learns one bounded smoothing
coefficient per channel by gradient descent, and repeats the final level over
the requested horizon.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `initial_alpha=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Exponential smoothing is a classical family of time series forecasting methods that assign exponentially decreasing weights to past observations, placing the most emphasis on recent data. Simple exponential smoothing forecasts a constant level, while double (Holt) and triple (Holt-Winters) variants additionally model additive or multiplicative trend and seasonality components via additional smoothing parameters. ExpSmoothingTS implements the core smoothing idea as a differentiable PyTorch module with learnable decay parameters, enabling the classical technique to be trained end-to-end with gradient descent and deployed on the same hardware as neural forecasting models.

## In ModernTSF
Default config: `configs/models/ExpSmoothingTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent implementation of the simple level-only exponential
smoothing recurrence; no external source implementation was inspected or
copied. It omits Holt trend and seasonal states, learns one bounded smoothing
coefficient per channel by gradient descent, and repeats the final level over
the requested horizon.

## Citation

```bibtex
@article{holt2004forecasting,
  author  = {Charles C. Holt},
  title   = {Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages},
  journal = {International Journal of Forecasting},
  volume  = {20},
  number  = {1},
  pages   = {5--10},
  year    = {2004},
  doi     = {10.1016/j.ijforecast.2003.09.015}
}
```
