---
name: "Crossformer"
summary: "Crossformer is a Transformer-based model for multivariate time series forecasting that explicitly models both temporal (cross-time) and inter-variable (cross-dimension) dependencies. It embeds the input series into a 2-D vector array via Dimension-Segment-Wise (DSW) embedding, applies a Two-Stage Attention (TSA) layer to efficiently capture both dependency types, and uses a Hierarchical Encoder-Decoder (HED) to leverage multi-scale temporal information for direct multi-step prediction."
paper: "https://openreview.net/forum?id=vSVLM2j9eie"
paper_title: "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting"
venue: "ICLR 2023"
year: 2023
code: "https://github.com/Thinklab-SJTU/Crossformer"
revision: "c10c8eadb153d1dd9798250967747ca3ebb81383"
license: "Apache-2.0"
---
# Crossformer

Crossformer is a Transformer-based model for multivariate time series forecasting that explicitly models both temporal (cross-time) and inter-variable (cross-dimension) dependencies. It embeds the input series into a 2-D vector array via Dimension-Segment-Wise (DSW) embedding, applies a Two-Stage Attention (TSA) layer to efficiently capture both dependency types, and uses a Hierarchical Encoder-Decoder (HED) to leverage multi-scale temporal information for direct multi-step prediction.

<!-- model-card:canonical:start -->
## Method overview

Crossformer is a Transformer-based model for multivariate time series forecasting that explicitly models both temporal (cross-time) and inter-variable (cross-dimension) dependencies.

## Core architecture

It embeds the input series into a 2-D vector array via Dimension-Segment-Wise (DSW) embedding, applies a Two-Stage Attention (TSA) layer to efficiently capture both dependency types, and uses a Hierarchical Encoder-Decoder (HED) to leverage multi-scale temporal information for direct multi-step prediction.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=vSVLM2j9eie); title: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting; venue/year: ICLR 2023 / 2023
- [codebase](https://github.com/Thinklab-SJTU/Crossformer); revision: `c10c8eadb153d1dd9798250967747ca3ebb81383`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/Crossformer.toml`](../../../../configs/models/Crossformer.toml).

## Differences

Clean-room implementation: confirmed.

Clean-room structure map: DSW segmentation; TSA temporal stage; router send/receive dimension stage; hierarchical segment merging; scale-wise direct forecast heads. The linked code is reference-only. The local direct multi-scale head is a disclosed simplification of HED.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `n_heads=4`, `e_layers=2`, `d_ff=128`, `seg_len=12`, `win_size=2`, `factor=10`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
- **Venue**: ICLR 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract
Recently many deep models have been proposed for multivariate time series (MTS) forecasting. In particular, Transformer-based models have shown great potential because they can capture long-term dependency. However, existing Transformer-based models mainly focus on modeling the temporal dependency (cross-time dependency) yet often omit the dependency among different variables (cross-dimension dependency), which is critical for MTS forecasting. To fill the gap, we propose Crossformer, a Transformer-based model utilizing cross-dimension dependency for MTS forecasting. In Crossformer, the input MTS is embedded into a 2D vector array through the Dimension-Segment-Wise (DSW) embedding to preserve time and dimension information. Then the Two-Stage Attention (TSA) layer is proposed to efficiently capture the cross-time and cross-dimension dependency. Utilizing DSW embedding and TSA layer, Crossformer establishes a Hierarchical Encoder-Decoder (HED) to use the information at different scales for the final forecasting. Extensive experimental results on six real-world datasets show the effectiveness of Crossformer against previous state-of-the-arts.

## In ModernTSF
Default config: `configs/models/Crossformer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed.

Clean-room structure map: DSW segmentation; TSA temporal stage; router send/receive dimension stage; hierarchical segment merging; scale-wise direct forecast heads. The linked code is reference-only. The local direct multi-scale head is a disclosed simplification of HED.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/ZhangY23,
  author       = {Yunhao Zhang and
                  Junchi Yan},
  title        = {Crossformer: Transformer Utilizing Cross-Dimension Dependency for
                  Multivariate Time Series Forecasting},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/forum?id=vSVLM2j9eie},
  timestamp    = {Wed, 24 Jul 2024 16:50:34 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/ZhangY23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
