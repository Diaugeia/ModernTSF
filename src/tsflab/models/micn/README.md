---
name: "MICN"
description: "Multi-scale isometric convolution network: multi-kernel seasonal-trend decomposition, then per-scale downsampled local and global convolutions with linear horizon maps. Use for long-term forecasting with trend plus local fluctuations at modest compute; not for calendar-driven targets (time marks are ignored)."
---

# MICN

## Idea

- `MultiScaleDecomposition` averages several moving-average decompositions (`series_decomposition`) into one seasonal and one trend component.
- Each `IsometricConvolutionBranch` downsamples with a strided depthwise conv (local pattern), adds a global context convolution over the shortened sequence, and restores length with a transposed conv; `MICLayer` merges the scales.
- The seasonal part is refined with a residual and mapped by a linear history-to-horizon layer; the trend part gets its own linear map and the two forecasts are summed.
- The input embedding is a linear map over channels; calendar embedding is omitted.

## When to use

- Designed for long-term forecasting where both local fluctuations and global trends matter, as a CNN alternative to quadratic attention (linear in sequence length).
- The explicit trend branch helps on trending series.
- Channels are mixed by the input embedding, so it is a weaker fit for many weakly correlated channels.
- No instance normalization and no calendar features.

## Configure

- `enc_in`: number of input channels.
- `conv_kernel`: distinct integers >= 2, smaller than `seq_len`; each is both a moving-average window (even values +1) and a downsampling stride (input replicate-padded to a multiple).

Other hyperparameters: preset defaults in `configs/models/MICN.toml`; tune generically.

## Differences

- Clean-room implementation from the paper; the unlicensed repository is reference-only.
- The global context step is a depthwise kernel-3 convolution plus a pointwise convolution over the downsampled sequence.
- Calendar embedding is intentionally omitted.
