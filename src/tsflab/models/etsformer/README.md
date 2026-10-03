---
name: "ETSformer"
description: "Exponential-smoothing Transformer: level/growth/seasonality decomposition via exponential-smoothing attention and top-k Fourier frequency attention. Use for long-term forecasting of series with clear trend and seasonality; not for calendar-driven or irregular series."
---

# ETSformer

## Idea

- `FrequencyAttention` keeps the top-k amplitude Fourier bases (excluding DC) of the residual and extrapolates them over history and horizon as seasonality.
- `ExponentialSmoothing` (exponential smoothing attention) smooths the first difference with a learnable per-feature alpha to extract growth.
- Each `ETSLayer` removes seasonality and growth from the residual in turn; damped growth (`damping_logit`) is extrapolated over the horizon.
- A level recurrence with learnable alpha adds the final level to the projected seasonality and growth forecasts; channels are embedded jointly by a circular convolution.

## When to use

- Designed for long-term forecasting where the series decomposes into level, growth and seasonality, giving interpretable components (classical exponential-smoothing principle).
- Seasonality is a few dominant Fourier bases, so it suits stable periodic patterns rather than shifting periods.
- Calendar covariates are deliberately not used.

## Configure

- `enc_in`: number of channels.

Other hyperparameters: preset defaults in `configs/models/ETSformer.toml`; tune generically.

## Differences

Paper-driven local implementation; `thuml/Time-Series-Library` at `230805fe` (MIT) is reference only, nothing copied.

- Implements the exponential-smoothing recurrence, top-amplitude Fourier seasonality, residual level/growth/seasonality stacks, growth damping and additive decoder.
- `n_heads`, `d_layers`, `embed` and `freq` are accepted for configuration compatibility but unused; timestamp marks are accepted and not consumed.
- No published-benchmark reproduction is claimed.

Citation: Woo, G., Liu, C., Sahoo, D., Kumar, A., Hoi, S. "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting." arXiv:2202.01381 (2022).
