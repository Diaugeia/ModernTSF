---
name: "SCINet"
description: "Recursive even/odd downsample-convolve-interact tree over the window, then a linear forecaster; convolutions mix all channels. Use for multi-resolution temporal structure that survives downsampling, on moderately many correlated channels; not for probabilistic output or windows shorter than 2**num_levels."
---

# SCINet

## Idea

- Temporal relations are largely preserved after downsampling a series into two sub-sequences; SCINet models time series by recursively downsampling, convolving, and interacting.
- `SCIInteraction` splits the sequence into even and odd subsequences and cross-scales each with `exp(phi(.))` and `exp(psi(.))` of a `TemporalOperator` convolution of the other, then updates with `rho` and `eta`.
- `SCITree` applies this recursively for `num_levels` (default 3) and `interleave`s the leaves back into time order.
- `SCINetStack` adds a residual, then `nn.Linear(seq_len, pred_len)`; up to three stacks are chained by feeding the prediction back into the history.
- Inputs are standardized per window and given a sinusoidal position encoding; the convolutions mix all channels.

## When to use

- Series with structure at several temporal resolutions; the tree combines features from every downsampled level.
- Correlated channels: the convolutions mix all channels, so width scales with the channel count; on many weakly correlated channels a channel-independent model is the simpler baseline.
- Point forecasts only; the window must hold at least `2**num_levels` steps.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `num_levels` follows `seq_len`: `seq_len >= 2**num_levels`.
- Other hyperparameters: preset defaults in `configs/models/SCINet.toml`; tune generically (`num_stacks <= 3`).

## Differences

Clean-room implementation of the published interaction equations, recursive node count, interleaving, and stacked forecast path; the linked Time-Series-Library implementation is reference-only and no code was copied.

- Inputs are `[B, seq_len, enc_in]`, outputs `[B, pred_len, enc_in]`; time marks are ignored.
- Intermediate supervision of stacked outputs is left to the trainer (not built in).

Citation: Liu, Zeng, Chen, Xu, Lai, Ma, Xu, "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction", NeurIPS 2022, arXiv:2106.09305.
