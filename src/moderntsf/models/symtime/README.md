---
name: "SymTime"
summary: "SymTime is a pre-trained time-series foundation model that leverages synthetic series-symbol data to overcome data scarcity and imbalance in time-series analysis. Drawing on complex dynamic system theories, it generates unlimited high-quality time-series data paired with symbolic expressions, then pre-trains a Transformer-based series encoder jointly with a symbol encoder (built on a pre-trained LLM) through masked time-series modelling and masked language modelling. The resulting representations are fine-tuned for downstream forecasting tasks, serving the standard multivariate time-series forecasting setting."
paper: "https://arxiv.org/abs/2510.08445"
paper_title: "Synthetic Series-Symbol Data Generation for Time Series Foundation Models"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/wwhenxuan/SymTime"
revision: "24a938649e140d92e8bf642d37a11fd9624e7698"
license: "MIT"
---
# SymTime

SymTime is a pre-trained time-series foundation model that leverages synthetic series-symbol data to overcome data scarcity and imbalance in time-series analysis. Drawing on complex dynamic system theories, it generates unlimited high-quality time-series data paired with symbolic expressions, then pre-trains a Transformer-based series encoder jointly with a symbol encoder (built on a pre-trained LLM) through masked time-series modelling and masked language modelling. The resulting representations are fine-tuned for downstream forecasting tasks, serving the standard multivariate time-series forecasting setting.

<!-- model-card:canonical:start -->
## Method overview

SymTime is a pre-trained time-series foundation model that leverages synthetic series-symbol data to overcome data scarcity and imbalance in time-series analysis.

## Core architecture

Drawing on complex dynamic system theories, it generates unlimited high-quality time-series data paired with symbolic expressions, then pre-trains a Transformer-based series encoder jointly with a symbol encoder (built on a pre-trained LLM) through masked time-series modelling and masked language modelling. The resulting representations are fine-tuned for downstream forecasting tasks, serving the standard multivariate time-series forecasting setting.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.08445); title: Synthetic Series-Symbol Data Generation for Time Series Foundation Models; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/wwhenxuan/SymTime); revision: `24a938649e140d92e8bf642d37a11fd9624e7698`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/SymTime.toml`](../../../../configs/models/SymTime.toml).

## Differences

Pinned source inspection: `models/pretrain_model.py`, `models/finetune_model.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

The local module is the downstream forecasting path, not a replacement for 50B-scale S² pre-training. It has no symbol/DistilBERT encoder, momentum encoders, MLM/MTM/contrastive objectives, or released pre-trained weights; the configurable compact Transformer defaults to two rather than six layers. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `patch_len=16`, `num_layers=2`, `num_heads=4`, `trend_kernel=25`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Synthetic Series-Symbol Data Generation for Time Series Foundation Models
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.08445

## Abstract
Foundation models for time series analysis (TSA) have attracted significant attention. However, challenges such as training data scarcity and imbalance continue to hinder their development. Inspired by complex dynamic system theories, we design a series-symbol data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic expressions. To leverage series-symbol data pairs with strong correlations, we develop SymTime, a pre-trained foundation model for enhancing time series representation using symbolic information. SymTime demonstrates competitive performance across five major TSA tasks when fine-tunes with downstream tasks, rivaling foundation models pre-trained on real-world datasets. This approach underscores the potential of series-symbol data generation and pretraining mechanisms in overcoming data scarcity and enhancing task performance. The code is available at https://github.com/wwhenxuan/SymTime.

## Source and verification

Pinned source inspection: `models/pretrain_model.py`, `models/finetune_model.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

The local module is the downstream forecasting path, not a replacement for 50B-scale S² pre-training. It has no symbol/DistilBERT encoder, momentum encoders, MLM/MTM/contrastive objectives, or released pre-trained weights; the configurable compact Transformer defaults to two rather than six layers. The reference-only repository was inspected at the pinned revision; no external source code was copied.

## In ModernTSF
Default config: `configs/models/SymTime.toml`; model specification: `spec.py`; clean-room forecasting implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2510-08445,
  author       = {Wenxuan Wang and
                  Kai Wu and
                  Yujian Betterest Li and
                  Dan Wang and
                  Xiaoyu Zhang},
  title        = {Synthetic Series-Symbol Data Generation for Time Series Foundation
                  Models},
  journal      = {CoRR},
  volume       = {abs/2510.08445},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.08445},
  doi          = {10.48550/ARXIV.2510.08445},
  eprinttype   = {arXiv},
  eprint       = {2510.08445},
  timestamp    = {Wed, 12 Nov 2025 07:27:09 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2510-08445.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
