---
name: "Sumba"
summary: "Sumba is a time series forecasting model for multivariate sequences that directly parameterizes spatial structures using a learnable matrix basis and a convex combination. Its dynamic spatial structure generation function operates within a well-constrained output space, producing lower-variance graph structures with interpretable dynamics, and combines dilated inception temporal convolution blocks with dynamic graph convolution to jointly model temporal dependencies and inter-variate correlations."
paper: "https://openreview.net/forum?id=co7DsOwcop"
paper_title: "Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/chenxiaodanhit/Sumba"
revision: "a1f8f45d2c89e4feb6c8e9399178c95157336f3b"
license: "NOASSERTION"
tagline: "Dynamic graph as a convex combination of learnable low-rank matrix bases, with multi-kernel gated temporal convs."
tags: ["cnn", "gnn", "graph-learning", "multi-scale", "normalization", "channel-mixing"]
composition: ["normalization=local:instance-mean-std-detached", "decomposition=none", "temporal=local:multi-kernel-gated-depthwise-conv", "channel=local:structured-matrix-basis-dynamic-graph-conv", "head=local:flatten-linear-head", "loss=loss:mse"]
---
# Sumba

## Key ideas

- `StructuredMatrixBasis` keeps a few learnable low-rank row-stochastic matrices and mixes them with softmax coefficients predicted from the pooled hidden state, so each sample's graph is a convex combination in a constrained space.
- `DynamicBasisGraphConv` propagates over that graph for several diffusion steps with a residual mix, then projects the concatenated steps.
- `MultiScaleTemporalConv` runs gated depthwise causal convs with kernels (2, 3, 5) in each pre-LayerNorm `SumbaBlock`.
- Each variable is z-scored over the lookback (statistics detached) and restored after a flatten linear head.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=co7DsOwcop); title: Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/chenxiaodanhit/Sumba); revision: `a1f8f45d2c89e4feb6c8e9399178c95157336f3b`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Sumba.toml`](../../../../configs/models/Sumba.toml).

## Differences

Clean-room implementation: confirmed. Paper mapping: structured parameterization → `StructuredMatrixBasis`; convex dynamic generation → its context-conditioned `forward`; temporal modeling → `MultiScaleTemporalConv`; spatial propagation → `DynamicBasisGraphConv`. The unlicensed reference is link-only and its source code was not copied. Dataset-specific regularization weights are not reproduced.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `basis_count=4`, `basis_rank=8`, `temporal_kernels=[2, 3, 5]`, `depth=2`, `diffusion_steps=2`, `mix=0.1`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics
- **Venue**: NeurIPS 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Multivariate time series forecasting is of central importance in modern intelligent decision systems. The dynamics of multivariate time series are jointly characterized by temporal dependencies and spatial correlations. Hence, it is equally important to build the forecasting models from both perspectives. The real-world multivariate time series data often presents spatial correlations that show structures and evolve dynamically. To capture such dynamic spatial structures, the existing forecasting approaches often rely on a two-stage learning process (learning dynamic series representations and then generating spatial structures), which is sensitive to the small time-window input data and has high variance. To address this, we propose a novel forecasting model with a structured matrix basis. At its core is a dynamic spatial structure generation function whose output space is well-constrained and the generated structures have lower variance, meanwhile, it is more expressive and can offer interpretable dynamics. This is achieved through a novel structured parameterization and imposing structure regularization on the matrix basis. Extensive experiments on six benchmark datasets demonstrate that our model achieves up to 8.5% improvements over the existing methods, while providing interpretability into the underlying system dynamics.

## In TSFLab
Default config: `configs/models/Sumba.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Paper mapping: structured parameterization → `StructuredMatrixBasis`; convex dynamic generation → its context-conditioned `forward`; temporal modeling → `MultiScaleTemporalConv`; spatial propagation → `DynamicBasisGraphConv`. The unlicensed reference is link-only and its source code was not copied. Dataset-specific regularization weights are not reproduced.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/ChenL0024,
  author       = {Xiaodan Chen and
                  Xiucheng Li and
                  Xinyang Chen and
                  Zhijun Li},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Structured Matrix Basis for Multivariate Time Series Forecasting with
                  Interpretable Dynamics},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/2b47305e1c81890b1089a405686ad183-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/ChenL0024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
