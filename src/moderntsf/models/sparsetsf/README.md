---
name: "SparseTSF"
summary: "SparseTSF is an extremely lightweight model for long-term time series forecasting that achieves competitive performance with fewer than 1,000 parameters. Its core innovation is the Cross-Period Sparse Forecasting technique, which decouples periodicity and trend by downsampling the original sequence so that the model focuses on cross-period trend prediction rather than point-wise temporal modelling."
paper: "https://arxiv.org/abs/2405.00946"
paper_title: "SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters"
venue: "ICML 2024"
year: 2024
code: "https://github.com/lss-1138/SparseTSF"
revision: "b8c2740eecc84d8095ffce49ba5acafe68e53bb8"
license: "Apache-2.0"
---
# SparseTSF

SparseTSF is an extremely lightweight model for long-term time series forecasting that achieves competitive performance with fewer than 1,000 parameters. Its core innovation is the Cross-Period Sparse Forecasting technique, which decouples periodicity and trend by downsampling the original sequence so that the model focuses on cross-period trend prediction rather than point-wise temporal modelling.

<!-- model-card:canonical:start -->
## Method overview

SparseTSF is an extremely lightweight model for long-term time series forecasting that achieves competitive performance with fewer than 1,000 parameters.

## Core architecture

Its core innovation is the Cross-Period Sparse Forecasting technique, which decouples periodicity and trend by downsampling the original sequence so that the model focuses on cross-period trend prediction rather than point-wise temporal modelling.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2405.00946); title: SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters; venue/year: ICML 2024 / 2024
- [codebase](https://github.com/lss-1138/SparseTSF); revision: `b8c2740eecc84d8095ffce49ba5acafe68e53bb8`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/SparseTSF.toml`](../../../../configs/models/SparseTSF.toml).

## Differences

**Paper-driven local implementation.** A shared learnable convolution aggregates within a period, mean-centered histories are reshaped into phase-aligned cross-period subsequences, one shared Linear/MLP forecasts each sparse subsequence, and the phases are interleaved back into time order. Non-divisible histories use the last complete periods and non-divisible horizons are trimmed from a final partial period. The external repository is reference-only and no source file was copied or adapted.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `period=24`, `d_model=64`, `model_type='linear'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters
- **Venue**: ICML 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.00946

## Abstract
This paper introduces SparseTSF, a novel, extremely lightweight model for Long-term Time Series Forecasting (LTSF), designed to address the challenges of modeling complex temporal dependencies over extended horizons with minimal computational resources. At the heart of SparseTSF lies the Cross-Period Sparse Forecasting technique, which simplifies the forecasting task by decoupling the periodicity and trend in time series data. This technique involves downsampling the original sequences to focus on cross-period trend prediction, effectively extracting periodic features while minimizing the model's complexity and parameter count. Based on this technique, the SparseTSF model uses fewer than 1k parameters to achieve competitive or superior performance compared to state-of-the-art models. Furthermore, SparseTSF showcases remarkable generalization capabilities, making it well-suited for scenarios with limited computational resources, small samples, or low-quality data. The code is publicly available at this repository.

## In ModernTSF
Default config: `configs/models/SparseTSF.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

**Paper-driven local implementation.** A shared learnable convolution aggregates within a period, mean-centered histories are reshaped into phase-aligned cross-period subsequences, one shared Linear/MLP forecasts each sparse subsequence, and the phases are interleaved back into time order. Non-divisible histories use the last complete periods and non-divisible horizons are trimmed from a final partial period. The external repository is reference-only and no source file was copied or adapted.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/Lin0WCY24,
  author       = {Shengsheng Lin and
                  Weiwei Lin and
                  Wentai Wu and
                  Haojun Chen and
                  Junjie Yang},
  editor       = {Ruslan Salakhutdinov and
                  Zico Kolter and
                  Katherine A. Heller and
                  Adrian Weller and
                  Nuria Oliver and
                  Jonathan Scarlett and
                  Felix Berkenkamp},
  title        = {SparseTSF: Modeling Long-term Time Series Forecasting with *1k* Parameters},
  booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024,
                  Vienna, Austria, July 21-27, 2024},
  series       = {Proceedings of Machine Learning Research},
  pages        = {30211--30226},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2024},
  url          = {https://proceedings.mlr.press/v235/lin24n.html},
  timestamp    = {Mon, 09 Feb 2026 17:23:53 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/Lin0WCY24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
