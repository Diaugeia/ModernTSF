---
name: "MultiPatchFormer"
summary: "MultiPatchFormer is a Transformer-based time series forecasting model that integrates multi-scale patch-wise temporal modeling with channel-wise representation learning. The input time series is divided into patches at multiple resolutions to capture temporal correlations across different time granularities; a subsequent channel-wise encoder models inter-series relationships; and a multi-step linear decoder generates the final multi-horizon predictions, reducing overfitting and noise effects. It targets both univariate and multivariate long-term forecasting settings."
paper: "https://doi.org/10.1038/s41598-024-82417-4"
paper_title: "A multiscale model for multivariate time series forecasting"
venue: "Scientific Reports 2025"
year: 2025
code: "https://github.com/thuml/Time-Series-Library"
revision: "4e938a1767106324dd753b2a44832bf870a0252e"
license: "MIT"
tagline: "Four patch scales fused into one token stream; temporal then channel Transformer layers; semi-autoregressive head."
tags: ["transformer", "patching", "multi-scale", "channel-mixing"]
composition: ["normalization=local:instance-standardization", "decomposition=none", "temporal=local:multi-scale-patch-transformer-encoder", "channel=local:channel-wise-transformer-layer", "head=local:semi-autoregressive-head", "loss=loss:mse"]
---
# MultiPatchFormer

## Key ideas

- `PatchScale` embeds patches of lengths 8, 16, 24, 32 (strides 8, 8, 7, 6); outputs are interpolated to a common token count and concatenated.
- Stacked `nn.TransformerEncoderLayer`s model temporal dependence per channel; the token mean then goes through one more encoder layer across channels.
- `SemiAutoregressiveHead` emits the horizon in groups, each linear layer also seeing the groups emitted before it.
- Series are standardized with per-window mean and standard deviation.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1038/s41598-024-82417-4); title: A multiscale model for multivariate time series forecasting; venue/year: Scientific Reports 2025 / 2025
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `4e938a1767106324dd753b2a44832bf870a0252e`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MultiPatchFormer.toml`](../../../../configs/models/MultiPatchFormer.toml).

## Differences

Clean-room implementation: confirmed.

Clean-room structure map: 8/16/24/32-point patch scales; aligned patch tokens; temporal Transformer stack; channel Transformer; progressive semi-autoregressive horizon groups. Reference links are provenance only and no source was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `n_heads=4`, `e_layers=2`, `d_ff=128`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A multiscale model for multivariate time series forecasting
- **Venue**: Scientific Reports 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Transformer based models for time-series forecasting have shown promising performance and during the past few years different Transformer variants have been proposed in time-series forecasting domain. However, most of the existing methods, mainly represent the time-series from a single scale, making it challenging to capture various time granularities or ignore inter-series correlations between the series which might lead to inaccurate forecasts. In this paper, we address the above mentioned shortcomings and propose a Transformer based model which integrates multi-scale patch-wise temporal modeling and channel-wise representation. In the multi-scale temporal part, the input time-series is divided into patches of different resolutions to capture temporal correlations associated with various scales. The channel-wise encoder which comes after the temporal encoder, models the relations among the input series to capture the intricate interactions between them. In our framework, we further design a multi-step linear decoder to generate the final predictions for the purpose of reducing over-fitting and noise effects. Extensive experiments on seven real world datasets indicate that our model (MultiPatchFormer) achieves state-of-the-art results by surpassing other current baseline models in terms of error metrics and shows stronger generalizability.

## In TSFLab
Default config: `configs/models/MultiPatchFormer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed.

Clean-room structure map: 8/16/24/32-point patch scales; aligned patch tokens; temporal Transformer stack; channel Transformer; progressive semi-autoregressive horizon groups. Reference links are provenance only and no source was copied.

## Citation

```bibtex
@article{naghashi2025multiscale,
  author  = {Vahid Naghashi and Mounir Boukadoum and Abdoulaye Banire Diallo},
  title   = {A Multiscale Model for Multivariate Time Series Forecasting},
  journal = {Scientific Reports},
  volume  = {15},
  number  = {1},
  year    = {2025},
  doi     = {10.1038/s41598-024-82417-4},
  url     = {https://doi.org/10.1038/s41598-024-82417-4}
}
```
