---
name: "FiLM"
summary: "FiLM is a standalone forecaster: each channel window is compressed by a Legendre-memory projection, filtered by a low-rank complex Fourier layer on the lowest rFFT modes, and reconstructed on the horizon grid; a learned mixture of multiscale experts combines the outputs inside RevIN. It is a full model here, not a plug-in module for other forecasters."
paper: "https://arxiv.org/abs/2205.08897"
paper_title: "FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting"
venue: "NeurIPS 2022"
year: 2022
code: "https://github.com/tianzhou2011/FiLM"
revision: "2794355ff6258743a29715263414283782910521"
license: "MIT"
tagline: "Legendre-memory projection of each window, low-rank complex Fourier filtering, and a mixture of multiscale experts."
tags: ["mlp", "frequency", "multi-scale", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:legendre-memory-and-low-rank-fourier-experts", "channel=local:channel-independent-shared-weights", "head=local:legendre-reconstruction-expert-mixture", "loss=loss:mse"]
---
# FiLM

## Key ideas

- `LegendreProjection` runs the translated-Legendre recurrence `C_t = A C_(t-1) + B x_t` (bilinear-discretized) to compress the history into polynomial coefficients.
- `LowRankFourierLayer` keeps the lowest rFFT modes of the coefficient trajectory and filters them with complex low-rank factors (`order`, `rank`).
- The last filtered state is reconstructed on the horizon grid by Legendre basis functions (`reconstruct`).
- `FiLMExpert`s read the last `scale * pred_len` steps (default scales 1, 2, 4) and are combined by a learned `expert_mixture`; `revin` wraps the model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2205.08897); title: FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting; venue/year: NeurIPS 2022 / 2022
- [codebase](https://github.com/tianzhou2011/FiLM); revision: `2794355ff6258743a29715263414283782910521`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/FiLM.toml`](../../../../configs/models/FiLM.toml).

## Differences

**Clean-room implementation: confirmed.** The linked MIT author repository is
`reference-only`; its source was not copied. The local design implements the
paper's Legendre state recurrence with torch-native bilinear discretization,
lowest-mode Fourier selection, complex low-rank factors, reconstruction, and a
mixture of multiscale history experts. The `order` and `rank` settings replace
the former ambiguous `window_size`. Random high-mode selection, integration as
a plug-in to other backbones, official initialization, checkpoint reference comparison, and
published-metric reference comparison are omitted.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `ratio=0.5`, `multiscale=[1, 2, 4]`, `order=64`, `rank=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting
- **Venue**: NeurIPS 2022
- **Published**: 2022 (arXiv: 2022-05)
- **arXiv**: https://arxiv.org/abs/2205.08897

## Abstract
Recent studies have shown that deep learning models such as RNNs and Transformers have brought significant performance gains for long-term forecasting of time series because they effectively utilize historical information. We found, however, that there is still great room for improvement in how to preserve historical information in neural networks while avoiding overfitting to noise present in the history. Addressing this allows better utilization of the capabilities of deep learning models. To this end, we design a Frequency improved Legendre Memory model, or FiLM: it applies Legendre polynomial projections to approximate historical information, uses Fourier projection to remove noise, and adds a low-rank approximation to speed up computation. Our empirical studies show that the proposed FiLM significantly improves the accuracy of state-of-the-art models in multivariate and univariate long-term forecasting by (19.2%, 22.6%), respectively. We also demonstrate that the representation module developed in this work can be used as a general plugin to improve the long-term prediction performance of other deep learning modules.

## In TSFLab
Default config: `configs/models/FiLM.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Source and verification

**Clean-room implementation: confirmed.** The linked MIT author repository is
`reference-only`; its source was not copied. The local design implements the
paper's Legendre state recurrence with torch-native bilinear discretization,
lowest-mode Fourier selection, complex low-rank factors, reconstruction, and a
mixture of multiscale history experts. The `order` and `rank` settings replace
the former ambiguous `window_size`. Random high-mode selection, integration as
a plug-in to other backbones, official initialization, checkpoint reference comparison, and
published-metric reference comparison are omitted.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/ZhouMWW0YY022,
  author       = {Tian Zhou and
                  Ziqing Ma and
                  Xue Wang and
                  Qingsong Wen and
                  Liang Sun and
                  Tao Yao and
                  Wotao Yin and
                  Rong Jin},
  editor       = {Sanmi Koyejo and
                  S. Mohamed and
                  A. Agarwal and
                  Danielle Belgrave and
                  K. Cho and
                  A. Oh},
  title        = {FiLM: Frequency improved Legendre Memory Model for Long-term Time
                  Series Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 35: Annual Conference
                  on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans,
                  LA, USA, November 28 - December 9, 2022},
  year         = {2022},
  url          = {http://papers.nips.cc/paper\_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract-Conference.html},
  timestamp    = {Thu, 23 Jan 2025 19:51:39 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/ZhouMWW0YY022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
