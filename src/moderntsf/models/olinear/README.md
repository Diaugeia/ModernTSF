---
name: "OLinear"
summary: "OLinear is a linear-based multivariate time series forecasting model that operates in an orthogonally transformed domain rather than directly in the time domain. It introduces OrthoTrans, a data-adaptive transformation built on an orthogonal matrix that diagonalizes the series' temporal Pearson correlation matrix via eigenvalue decomposition, yielding a decorrelated feature space for linear encoding. Complementing this, OLinear uses NormLin, a customized linear layer with a normalized weight matrix to capture multivariate dependencies, which empirically outperforms multi-head self-attention while requiring roughly half the FLOPs."
paper: "https://arxiv.org/abs/2505.08550"
paper_title: "OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/jackyue1994/OLinear"
revision: "f168e01a3e0e316ad98330b5e77afed1f77b0af5"
license: "NOASSERTION"
---
# OLinear

OLinear is a linear-based multivariate time series forecasting model that operates in an orthogonally transformed domain rather than directly in the time domain. It introduces OrthoTrans, a data-adaptive transformation built on an orthogonal matrix that diagonalizes the series' temporal Pearson correlation matrix via eigenvalue decomposition, yielding a decorrelated feature space for linear encoding. Complementing this, OLinear uses NormLin, a customized linear layer with a normalized weight matrix to capture multivariate dependencies, which empirically outperforms multi-head self-attention while requiring roughly half the FLOPs.

<!-- model-card:canonical:start -->
## Method overview

OLinear is a linear-based multivariate time series forecasting model that operates in an orthogonally transformed domain rather than directly in the time domain.

## Core architecture

It introduces OrthoTrans, a data-adaptive transformation built on an orthogonal matrix that diagonalizes the series' temporal Pearson correlation matrix via eigenvalue decomposition, yielding a decorrelated feature space for linear encoding. Complementing this, OLinear uses NormLin, a customized linear layer with a normalized weight matrix to capture multivariate dependencies, which empirically outperforms multi-head self-attention while requiring roughly half the FLOPs.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.08550); title: OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/jackyue1994/OLinear); revision: `f168e01a3e0e316ad98330b5e77afed1f77b0af5`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/OLinear.toml`](../../../../configs/models/OLinear.toml).

## Differences

Clean-room implementation: confirmed.

This is an independent implementation derived from equations (2)--(5) of the
paper. `Model.input_basis` and `Model.output_basis` implement OrthoTrans,
`NormLin.normalized_weight()` implements the positive row-wise L1-normalized
matrix in equation (3), and the residual channel/sequence learners correspond
to CSL and ISL. The linked repository is reference-only; its source was not
inspected or copied.

The paper computes both orthogonal bases once from training-set lag-correlation
matrices. The standalone preset cannot infer those dataset-level statistics, so
it serializes identity bases by default and exposes `set_orthogonal_bases()` for
installing externally computed eigenvectors. This compact rewrite uses one
CSL/ISL block and a direct flattened decoder rather than reproducing the full
training harness or paper hyperparameter sweep.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `dropout=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.08550

## Abstract
This paper presents OLinear, a linear-based multivariate time series forecasting model that operates in an orthogonally transformed domain. Recent forecasting models typically adopt the temporal forecast (TF) paradigm, which directly encode and decode time series in the time domain. However, the entangled step-wise dependencies in series data can hinder the performance of TF. To address this, some forecasters conduct encoding and decoding in the transformed domain using fixed, dataset-independent bases (e.g., sine and cosine signals in the Fourier transform). In contrast, we utilize OrthoTrans, a data-adaptive transformation based on an orthogonal matrix that diagonalizes the series' temporal Pearson correlation matrix. This approach enables more effective encoding and decoding in the decorrelated feature domain and can serve as a plug-in module to enhance existing forecasters. To enhance the representation learning for multivariate time series, we introduce a customized linear layer, NormLin, which employs a normalized weight matrix to capture multivariate dependencies. Empirically, the NormLin module shows a surprising performance advantage over multi-head self-attention, while requiring nearly half the FLOPs. Extensive experiments on 24 benchmarks and 140 forecasting tasks demonstrate that OLinear consistently achieves state-of-the-art performance with high efficiency. Notably, as a plug-in replacement for self-attention, the NormLin module consistently enhances Transformer-based forecasters.

## Source and verification

Clean-room implementation: confirmed.

This is an independent implementation derived from equations (2)--(5) of the
paper. `Model.input_basis` and `Model.output_basis` implement OrthoTrans,
`NormLin.normalized_weight()` implements the positive row-wise L1-normalized
matrix in equation (3), and the residual channel/sequence learners correspond
to CSL and ISL. The linked repository is reference-only; its source was not
inspected or copied.

The paper computes both orthogonal bases once from training-set lag-correlation
matrices. The standalone preset cannot infer those dataset-level statistics, so
it serializes identity bases by default and exposes `set_orthogonal_bases()` for
installing externally computed eigenvectors. This compact rewrite uses one
CSL/ISL block and a direct flattened decoder rather than reproducing the full
training harness or paper hyperparameter sweep.

## In ModernTSF
Default config: `configs/models/OLinear.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2505-08550,
  author       = {Wenzhen Yue and
                  Yong Liu and
                  Haoxuan Li and
                  Hao Wang and
                  Xianghua Ying and
                  Ruohao Guo and
                  Bowei Xing and
                  Ji Shi},
  title        = {OLinear: {A} Linear Model for Time Series Forecasting in Orthogonally
                  Transformed Domain},
  journal      = {CoRR},
  volume       = {abs/2505.08550},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.08550},
  doi          = {10.48550/ARXIV.2505.08550},
  eprinttype   = {arXiv},
  eprint       = {2505.08550},
  timestamp    = {Fri, 08 May 2026 07:40:48 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-08550.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
