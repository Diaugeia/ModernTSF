---
name: "LightTS"
summary: "LightTS is a lightweight MLP-based model for multivariate time-series forecasting. It applies simple MLP structures on top of two complementary down-sampling strategies — interval sampling and continuous sampling — to efficiently capture temporal patterns while using a fraction of the compute required by Transformer or RNN-based approaches."
paper: "https://arxiv.org/abs/2207.01186"
paper_title: "Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures"
venue: "arXiv preprint"
year: 2022
code: "https://github.com/d-gcc/LightTS"
revision: "362ca172791559766f6a055be8f2cbed1bad5530"
license: "NOASSERTION"
---
# LightTS

LightTS is a lightweight MLP-based model for multivariate time-series forecasting. It applies simple MLP structures on top of two complementary down-sampling strategies — interval sampling and continuous sampling — to efficiently capture temporal patterns while using a fraction of the compute required by Transformer or RNN-based approaches.

<!-- model-card:canonical:start -->
## Method overview

LightTS is a lightweight MLP-based model for multivariate time-series forecasting.

## Core architecture

It applies simple MLP structures on top of two complementary down-sampling strategies — interval sampling and continuous sampling — to efficiently capture temporal patterns while using a fraction of the compute required by Transformer or RNN-based approaches.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2207.01186); title: Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures; venue/year: arXiv preprint / 2022
- [codebase](https://github.com/d-gcc/LightTS); revision: `362ca172791559766f6a055be8f2cbed1bad5530`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/LightTS.toml`](../../../../configs/models/LightTS.toml).

## Differences

**Clean-room implementation: confirmed.** Continuous and interval sampling map
Equations 1--2 for divisible lengths; the three IEBlocks map the bottleneck
information-exchange procedure. Reference-only source was not copied and
published-metric reference comparison is not claimed.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `hid_dim=128`, `dropout=0.0`, `chunk_size=24`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures
- **Venue**: arXiv preprint
- **Published**: 2022
- **arXiv**: https://arxiv.org/abs/2207.01186

## Abstract
Multivariate time series forecasting has seen widely ranging applications in various domains, including finance, traffic, energy, and healthcare. To capture the sophisticated temporal patterns, plenty of research studies designed complex neural network architectures based on many variants of RNNs, GNNs, and Transformers. However, complex models are often computationally expensive and thus face a severe challenge in training and inference efficiency when applied to large-scale real-world datasets. In this paper, we introduce LightTS, a light deep learning architecture merely based on simple MLP-based structures. The key idea of LightTS is to apply an MLP-based structure on top of two delicate down-sampling strategies, including interval sampling and continuous sampling, inspired by a crucial fact that down-sampling time series often preserves the majority of its information. We conduct extensive experiments on eight widely used benchmark datasets. Compared with the existing state-of-the-art methods, LightTS demonstrates better performance on five of them and comparable performance on the rest. Moreover, LightTS is highly efficient. It uses less than 5% FLOPS compared with previous SOTA methods on the largest benchmark dataset. In addition, LightTS is robust and has a much smaller variance in forecasting accuracy than previous SOTA methods in long sequence forecasting tasks.

## In ModernTSF
Default config: `configs/models/LightTS.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`.

## Source and verification

**Clean-room implementation: confirmed.** Continuous and interval sampling map
Equations 1--2 for divisible lengths; the three IEBlocks map the bottleneck
information-exchange procedure. Reference-only source was not copied and
published-metric reference comparison is not claimed.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2207-01186,
  author       = {Tianping Zhang and
                  Yizhuo Zhang and
                  Wei Cao and
                  Jiang Bian and
                  Xiaohan Yi and
                  Shun Zheng and
                  Jian Li},
  title        = {Less Is More: Fast Multivariate Time Series Forecasting with Light
                  Sampling-oriented {MLP} Structures},
  journal      = {CoRR},
  volume       = {abs/2207.01186},
  year         = {2022},
  url          = {https://doi.org/10.48550/arXiv.2207.01186},
  doi          = {10.48550/ARXIV.2207.01186},
  eprinttype   = {arXiv},
  eprint       = {2207.01186},
  timestamp    = {Mon, 16 Jun 2025 17:44:15 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2207-01186.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
