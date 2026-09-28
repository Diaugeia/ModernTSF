---
name: "HN_MVTS"
summary: "HN_MVTS integrates a hypernetwork-based generative prior with any base neural-network forecaster for multivariate time-series forecasting. The hypernetwork takes a learnable embedding matrix of time-series components as input and generates the weights of the base model's final layer, acting as a data-adaptive regulariser that improves generalisation and long-range predictive accuracy — used only during training so it adds no inference overhead. This approach bridges the gap between high-accuracy channel-dependent models and the robustness of channel-independent models."
paper: "https://arxiv.org/abs/2511.08340"
paper_title: "HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/av-savchenko/HN-MVTS"
revision: "e86c58a315576cef021d99e04b9b5fef55ddd6d6"
license: "Apache-2.0"
---
# HN_MVTS

HN_MVTS integrates a hypernetwork-based generative prior with any base neural-network forecaster for multivariate time-series forecasting. The hypernetwork takes a learnable embedding matrix of time-series components as input and generates the weights of the base model's final layer, acting as a data-adaptive regulariser that improves generalisation and long-range predictive accuracy — used only during training so it adds no inference overhead. This approach bridges the gap between high-accuracy channel-dependent models and the robustness of channel-independent models.

<!-- model-card:canonical:start -->
## Method overview

HN_MVTS integrates a hypernetwork-based generative prior with any base neural-network forecaster for multivariate time-series forecasting.

## Core architecture

The hypernetwork takes a learnable embedding matrix of time-series components as input and generates the weights of the base model's final layer, acting as a data-adaptive regulariser that improves generalisation and long-range predictive accuracy — used only during training so it adds no inference overhead. This approach bridges the gap between high-accuracy channel-dependent models and the robustness of channel-independent models.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2511.08340); title: HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/av-savchenko/HN-MVTS); revision: `e86c58a315576cef021d99e04b9b5fef55ddd6d6`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/HN_MVTS.toml`](../../../../configs/models/HN_MVTS.toml).

## Differences

Pinned source inspection: `src/layers.py`, `src/moderntsf/models/dlinear.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This local implementation follows equations (2)–(4): a learnable embedding
for each channel is passed through a one-hidden-layer hypernetwork to generate
that channel's final projection weights. A compact channel-independent temporal
MLP is the local base model. Embeddings are learned from random initialization
rather than initialized with training-split Pearson/PCA statistics, and the
paper's alternative PatchTST/TSMixer backbones are not included. The
reference-only repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `embedding_dim=8`, `hyper_hidden=32`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-11)
- **arXiv**: https://arxiv.org/abs/2511.08340

## Abstract
Accurate forecasting of multivariate time series data remains a formidable challenge, particularly due to the growing complexity of temporal dependencies in real-world scenarios. While neural network-based models have achieved notable success in this domain, complex channel-dependent models often suffer from performance degradation compared to channel-independent models that do not consider the relationship between components but provide high robustness due to small capacity. In this work, we propose HN-MVTS, a novel architecture that integrates a hypernetwork-based generative prior with an arbitrary neural network forecasting model. The input of this hypernetwork is a learnable embedding matrix of time series components. To restrict the number of new parameters, the hypernetwork learns to generate the weights of the last layer of the target forecasting networks, serving as a data-adaptive regularizer that improves generalization and long-range predictive accuracy. The hypernetwork is used only during the training, so it does not increase the inference time compared to the base forecasting model. Extensive experiments on eight benchmark datasets demonstrate that application of HN-MVTS to the state-of-the-art models (DLinear, PatchTST, TSMixer, etc.) typically improves their performance. Our findings suggest that hypernetwork-driven parameterization offers a promising direction for enhancing existing forecasting techniques in complex scenarios.

## Source and verification

Pinned source inspection: `src/layers.py`, `src/moderntsf/models/dlinear.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed.

This local implementation follows equations (2)–(4): a learnable embedding
for each channel is passed through a one-hidden-layer hypernetwork to generate
that channel's final projection weights. A compact channel-independent temporal
MLP is the local base model. Embeddings are learned from random initialization
rather than initialized with training-split Pearson/PCA statistics, and the
paper's alternative PatchTST/TSMixer backbones are not included. The
reference-only repository was inspected at the pinned revision; no external source code was copied.

## In ModernTSF
Default config: `configs/models/HN_MVTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/SavchenkoK26,
  author       = {Andrey V. Savchenko and
                  Oleg Kachan},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {{HN-MVTS:} HyperNetwork-based Multivariate Time Series Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25200--25208},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39711},
  doi          = {10.1609/AAAI.V40I30.39711},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/SavchenkoK26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
