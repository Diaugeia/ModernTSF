---
name: "Pathformer"
summary: "Pathformer is a multi-scale Transformer for multivariate time-series forecasting that integrates temporal resolution and temporal distance in a unified framework. It divides the input series into patches of multiple sizes (multi-scale division), applies dual attention over each scale to capture both global correlations and local details, and routes the information through adaptive pathways that dynamically adjust the multi-scale modelling process based on the varying temporal dynamics of each input."
paper: "https://arxiv.org/abs/2402.05956"
paper_title: "Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting"
venue: "ICLR 2024"
year: 2024
code: "https://github.com/decisionintelligence/pathformer"
revision: "ea85d82932215e171357da47b3bc82d502344758"
license: "NOASSERTION"
tagline: "Multi-scale Transformer with dual (local and global) patch attention experts mixed by an input-dependent router."
tags: ["transformer", "multi-scale", "patching", "mixture-of-experts", "attention-variant", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:adaptive-multi-scale-pathway", "channel=local:channel-independent-shared-weights", "head=local:linear-over-time-head", "loss=loss:mse"]
---
# Pathformer

## Key ideas

- `DualScaleAttention` embeds samples, applies local attention inside each patch, then global attention across patches, for one patch size.
- `AdaptivePathway` holds one such expert per patch size and mixes them with a router fed by mean, standard deviation, mean absolute difference, and spectral concentration.
- Routing is dense and differentiable; the top-k pathways are only recorded in `last_topk` for inspection.
- Pathway layers are residual with LayerNorm, followed by `nn.Linear(seq_len, pred_len)`; `revin` wraps the model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.05956); title: Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/decisionintelligence/pathformer); revision: `ea85d82932215e171357da47b3bc82d502344758`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Pathformer.toml`](../../../../configs/models/Pathformer.toml).

## Differences

Clean-room implementation: confirmed. The implementation was derived independently from the paper's multi-scale division, dual-attention, and adaptive-pathway descriptions; source from the unlicensed reference repository was not copied or reused. Dense differentiable routing replaces hard sparse dispatch because the common forecast API has no auxiliary balance-loss channel; inspectable top-k paths are retained.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `layer_nums=3`, `k=2`, `num_experts=4`, `patch_size_list=[16, 12, 8, 6, 16, 12, 8, 6, 16, 12, 8, 6]`, `d_model=16`, `d_ff=64`, `residual_connection=1`, `revin=True`, `n_heads=4`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.05956

## Abstract
Transformers for time series forecasting mainly model time series from limited or fixed scales, making it challenging to capture different characteristics spanning various scales. We propose Pathformer, a multi-scale Transformer with adaptive pathways. It integrates both temporal resolution and temporal distance for multi-scale modeling. Multi-scale division divides the time series into different temporal resolutions using patches of various sizes. Based on the division of each scale, dual attention is performed over these patches to capture global correlations and local details as temporal dependencies. We further enrich the multi-scale Transformer with adaptive pathways, which adaptively adjust the multi-scale modeling process based on the varying temporal dynamics of the input, improving the accuracy and generalization of Pathformer. Extensive experiments on eleven real-world datasets demonstrate that Pathformer not only achieves state-of-the-art performance by surpassing all current models but also exhibits stronger generalization abilities under various transfer scenarios.

## In TSFLab
Default config: `configs/models/Pathformer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. The implementation was derived independently from the paper's multi-scale division, dual-attention, and adaptive-pathway descriptions; source from the unlicensed reference repository was not copied or reused. Dense differentiable routing replaces hard sparse dispatch because the common forecast API has no auxiliary balance-loss channel; inspectable top-k paths are retained.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/ChenZ0SWW0G24,
  author       = {Peng Chen and
                  Yingying Zhang and
                  Yunyao Cheng and
                  Yang Shu and
                  Yihang Wang and
                  Qingsong Wen and
                  Bin Yang and
                  Chenjuan Guo},
  title        = {Pathformer: Multi-scale Transformers with Adaptive Pathways for Time
                  Series Forecasting},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=lJkOCMP2aW},
  timestamp    = {Tue, 12 Aug 2025 11:51:29 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/ChenZ0SWW0G24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
