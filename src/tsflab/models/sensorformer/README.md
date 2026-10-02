---
name: "Sensorformer"
summary: "Sensorformer patches each variable's window the way PatchTST does, then refines the patch representations with a stack of two-stage Sensor Attention Blocks: stage one compresses every variable's patches into one 'Sensor' summary token via cross-attention (using the variable's last patch as query and all variables' patches as key/value), and stage two lets every patch attend back to those summaries, jointly extracting cross-variable and cross-time dependencies at a fraction of pure cross-patch self-attention's cost; a shared linear head then maps each variable's flattened final patch representations to the forecast horizon."
paper: "https://arxiv.org/abs/2501.03284"
paper_title: "Sensorformer: Cross-patch attention with global-patch compression is effective for high-dimensional multivariate time series forecasting"
venue: "arXiv preprint"
year: 2025

tagline: "PatchTST-style patches with two-stage sensor attention: per-variable summary tokens, then patches attend back."
tags: ["transformer", "patching", "attention-variant", "channel-mixing", "sparse-attention"]
composition: ["normalization=none", "decomposition=none", "temporal=component:embed+component:global_patch_compression_attention", "channel=component:global_patch_compression_attention", "head=component:flatten_forecast_head", "loss=loss:mse"]
---
# Sensorformer

## Key ideas

- `PatchEmbedding` (`embed`) replicate-pads by `stride`, extracts overlapping patches per variable, and embeds them with positions.
- `GlobalPatchCompressionAttention` stage one uses each variable's last patch as the query over all variables' patches to build one Sensor summary token per variable.
- Stage two lets every patch of every variable attend to those summaries, capturing cross-variable and cross-time dependence without full quadratic cross-patch attention.
- `FlattenForecastHead` maps each variable's final patches to the horizon; there is no instance normalization.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2501.03284); title: Sensorformer: Cross-patch attention with global-patch compression is effective for high-dimensional multivariate time series forecasting; venue/year: arXiv preprint / 2025
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Sensorformer.toml`](../../../../configs/models/Sensorformer.toml).

## Differences

**No official code was available.** The repository linked by the paper
(https://github.com/BigYellowTiger/Sensorformer) is an empty GitHub
repository — verified via the GitHub API: zero commits, and a contents
request returns `"Git Repository is empty."` No revision could be pinned.

**Paper-only implementation.** This model is implemented directly from the
paper's Section 3 text, Fig. 2, and the pseudocode in Algorithms 1-2, which
fully specify the patching scheme (explicitly "the same patch strategy as
PatchTST"), the two-stage attention's queries/keys/values, and the
`LayerNorm`/MLP wrapping of each stage. The paper does not specify patch
length, stride, `d_model`, number of heads, feed-forward width, number of
layers, or normalization before patching; this implementation exposes them as
ordinary parameters with PatchTST-style defaults rather than guessing
paper-specific values that were never published. `reference_comparison` is
declared not-applicable in `verification/models.toml` because there is no
official implementation to compare against.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`global_patch_compression_attention`](../_components/global_patch_compression_attention/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `n_heads=4`, `d_ff=128`, `layers=2`, `patch_len=16`, `stride=8`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Source and verification

**No official code was available.** The repository linked by the paper
(https://github.com/BigYellowTiger/Sensorformer) is an empty GitHub
repository — verified via the GitHub API: zero commits, and a contents
request returns `"Git Repository is empty."` No revision could be pinned.

**Paper-only implementation.** This model is implemented directly from the
paper's Section 3 text, Fig. 2, and the pseudocode in Algorithms 1-2, which
fully specify the patching scheme (explicitly "the same patch strategy as
PatchTST"), the two-stage attention's queries/keys/values, and the
`LayerNorm`/MLP wrapping of each stage. The paper does not specify patch
length, stride, `d_model`, number of heads, feed-forward width, number of
layers, or normalization before patching; this implementation exposes them as
ordinary parameters with PatchTST-style defaults rather than guessing
paper-specific values that were never published. `reference_comparison` is
declared not-applicable in `verification/models.toml` because there is no
official implementation to compare against.
