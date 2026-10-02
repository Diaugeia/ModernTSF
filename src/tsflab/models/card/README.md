---
name: "CARD"
summary: "CARD (Channel Aligned Robust Blend Transformer) is a Transformer-based model for multivariate long-term and short-term time series forecasting. It addresses the limitations of channel-independent Transformers by introducing a channel-aligned attention structure that jointly captures temporal correlations and cross-variable dependencies, a token blend module for multi-scale feature extraction, and a robust uncertainty-weighted loss function to reduce overfitting."
paper: "https://arxiv.org/abs/2305.12095"
paper_title: "CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting"
venue: "ICLR 2024"
year: 2024
code: "https://github.com/wxie9/CARD"
revision: "ca6d34bcf26355bfdb6fc05f49c66e7601817f66"
license: "NOASSERTION"
tagline: "Patch Transformer that blends EMA-aligned temporal-token attention with cross-channel attention over aligned tokens."
tags: ["transformer", "patching", "channel-mixing", "attention-variant"]
composition: ["normalization=none", "decomposition=none", "temporal=local:ema-aligned-patch-token-attention", "channel=local:channel-aligned-cross-channel-attention", "head=local:flatten-linear-head", "loss=loss:mse"]
---
# CARD

## Key ideas

- `ChannelAlignedBlock` runs two attentions per layer: over patch tokens of each channel (queries/keys from `exponential_smooth`, a causal EMA over tokens) and across channels at each patch position.
- A learned sigmoid gate (`blend`) mixes the temporal and cross-channel outputs token by token.
- Overlapping patches (length 16, stride 8) are linearly projected with learned positions; an optional statistics token (`use_statistic`) is available.
- A flatten linear head forecasts all channels; the paper's robust training loss and low-rank attention approximation are omitted.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2305.12095); title: CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/wxie9/CARD); revision: `ca6d34bcf26355bfdb6fc05f49c66e7601817f66`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CARD.toml`](../../../../configs/models/CARD.toml).

## Differences

Clean-room implementation: confirmed.

Clean-room structure map: patch projection; causal EMA token alignment; temporal-token and cross-channel attention; learned token blend; direct forecast head. The reference repository is retained only as a provenance link and none of its source was copied. Differences are the omitted robust training loss and low-rank attention approximation.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `dropout=0.1`, `alpha=0.5`, `use_statistic=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.12095

## Abstract
Recent studies have demonstrated the great power of Transformer models for time series forecasting. One of the key elements that lead to the transformer's success is the channel-independent (CI) strategy to improve the training robustness. However, the ignorance of the correlation among different channels in CI would limit the model's forecasting capacity. In this work, we design a special Transformer, i.e., Channel Aligned Robust Blend Transformer (CARD for short), that addresses key shortcomings of CI type Transformer in time series forecasting. First, CARD introduces a channel-aligned attention structure that allows it to capture both temporal correlations among signals and dynamical dependence among multiple variables over time. Second, in order to efficiently utilize the multi-scale knowledge, we design a token blend module to generate tokens with different resolutions. Third, we introduce a robust loss function for time series forecasting to alleviate the potential overfitting issue. This new loss function weights the importance of forecasting over a finite horizon based on prediction uncertainties. Our evaluation of multiple long-term and short-term forecasting datasets demonstrates that CARD significantly outperforms state-of-the-art time series forecasting methods. The code is available at the following repository: https://github.com/wxie9/CARD

## In TSFLab
Default config: `configs/models/CARD.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed.

Clean-room structure map: patch projection; causal EMA token alignment; temporal-token and cross-channel attention; learned token blend; direct forecast head. The reference repository is retained only as a provenance link and none of its source was copied. Differences are the omitted robust training loss and low-rank attention approximation.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/WangZWGD024,
  author       = {Xue Wang and
                  Tian Zhou and
                  Qingsong Wen and
                  Jinyang Gao and
                  Bolin Ding and
                  Rong Jin},
  title        = {{CARD:} Channel Aligned Robust Blend Transformer for Time Series Forecasting},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=MJksrOhurE},
  timestamp    = {Thu, 23 Jan 2025 19:51:39 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/WangZWGD024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
