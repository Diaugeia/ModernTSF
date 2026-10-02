---
name: "SOFTS"
summary: "SOFTS (Series-cOre Fused Time Series forecaster) is an MLP-based model for multivariate time-series forecasting in the standard time-series setting. Its key innovation is the STar Aggregate-Redistribute (STAR) module, which uses a centralized strategy to model inter-channel dependencies: all series are aggregated into a single global core representation, which is then fused back with each individual series, achieving linear-complexity channel interaction without relying on distributed attention mechanisms."
paper: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html"
paper_title: "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/Secilia-Cxy/SOFTS"
revision: "f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44"
license: "MIT"
---
# SOFTS

SOFTS (Series-cOre Fused Time Series forecaster) is an MLP-based model for multivariate time-series forecasting in the standard time-series setting. Its key innovation is the STar Aggregate-Redistribute (STAR) module, which uses a centralized strategy to model inter-channel dependencies: all series are aggregated into a single global core representation, which is then fused back with each individual series, achieving linear-complexity channel interaction without relying on distributed attention mechanisms.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html); title: SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/Secilia-Cxy/SOFTS); revision: `f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SOFTS.toml`](../../../../configs/models/SOFTS.toml).

## Differences

Clean-room implementation: confirmed. `SeriesCoreFusion.aggregate` implements centralized series-to-core aggregation and `forward` implements core redistribution in linear channel complexity. Reference-only source code was not copied; this forecast-only rewrite does not claim numerical reference comparison.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_core=64`, `d_ff=256`, `e_layers=2`, `dropout=0.1`, `activation='gelu'`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-04)
- **arXiv**: https://arxiv.org/abs/2404.14197

## Abstract
Multivariate time series forecasting plays a crucial role in various fields such as finance, traffic management, energy, and healthcare. Recent studies have highlighted the advantages of channel independence to resist distribution drift but neglect channel correlations, limiting further enhancements. Several methods utilize mechanisms like attention or mixer to address this by capturing channel correlations, but they either introduce excessive complexity or rely too heavily on the correlation to achieve satisfactory results under distribution drifts, particularly with a large number of channels. Addressing this gap, this paper presents an efficient MLP-based model, the Series-cOre Fused Time Series forecaster (SOFTS), which incorporates a novel STar Aggregate-Redistribute (STAR) module. Unlike traditional approaches that manage channel interactions through distributed structures, e.g., attention, STAR employs a centralized strategy to improve efficiency and reduce reliance on the quality of each channel. It aggregates all series to form a global core representation, which is then dispatched and fused with individual series representations to facilitate channel interactions effectively. SOFTS achieves superior performance over existing state-of-the-art methods with only linear complexity. The broad applicability of the STAR module across different forecasting models is also demonstrated empirically. For further research and development, we have made our code publicly available.

## In ModernTSF
Default config: `configs/models/SOFTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. `SeriesCoreFusion.aggregate` implements centralized series-to-core aggregation and `forward` implements core redistribution in linear channel complexity. Reference-only source code was not copied; this forecast-only rewrite does not claim numerical reference comparison.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LuCYZ24,
  author       = {Lu Han and
                  Xu{-}Yang Chen and
                  Han{-}Jia Ye and
                  De{-}Chuan Zhan},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {{SOFTS:} Efficient Multivariate Time Series Forecasting with Series-Core
                  Fusion},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/LuCYZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
