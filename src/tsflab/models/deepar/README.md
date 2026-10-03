---
name: "DeepAR"
description: "Global autoregressive LSTM shared across channels that emits a Gaussian per step and feeds back its own mean. Use for probabilistic forecasting of many related, roughly independent series; not for exploiting cross-channel dependence or very long horizons."
---

# DeepAR

## Idea

- One global LSTM is shared by all channels, each treated as its own series; inputs are the scalar value plus optional time-mark covariates.
- `gaussian_parameter_head` outputs location and positive scale per step; the model returns `[batch, pred_len, channels, 2]` (output type `distribution`).
- Decoding is autoregressive: after encoding history, the likelihood mean is fed back as the next input.
- Trained with Gaussian negative log-likelihood (`nll_gaussian`).

## When to use

- Probabilistic forecasts (mean and scale) for many related series that share dynamics, such as demand across items.
- Channels are modelled independently by one shared network; cross-channel correlation is not used.
- Step-by-step decoding accumulates error on long horizons and is slower than direct heads.

## Configure

- `enc_in`: the dataset's channel count.
- `cov_feat_size`: number of time-mark features fed as covariates (at most the mark width; shorter marks are zero-padded); `0` disables covariates.

Other hyperparameters: preset defaults in `configs/models/DeepAR.toml`; tune generically.

## Differences

- Clean-room implementation of the autoregressive likelihood, recurrent transition and Gaussian parameterization; the Apache-2.0 BasicTS code is reference only.
- Mean feedback replaces ancestral sampling, so the output is one Gaussian per step, not sample paths.
- Published-metric and checkpoint reference comparison are not claimed.
