---
name: "TimeBridge"
summary: "TimeBridge is a patch-based Transformer framework for multivariate long-term time-series forecasting that explicitly handles non-stationarity at two granularities: Integrated Attention removes short-term non-stationarity within each variate's patches to capture stable local dependencies, while Cointegrated Attention preserves non-stationarity across variates to model long-term cointegration relationships between channels."
paper: "https://arxiv.org/abs/2410.04442"
paper_title: "TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting"
venue: "ICML 2025"
year: 2025
code: "https://github.com/Hank0626/TimeBridge"
revision: "0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b"
license: "MIT"
---
# TimeBridge

TimeBridge is a patch-based Transformer framework for multivariate long-term time-series forecasting that explicitly handles non-stationarity at two granularities: Integrated Attention removes short-term non-stationarity within each variate's patches to capture stable local dependencies, while Cointegrated Attention preserves non-stationarity across variates to model long-term cointegration relationships between channels.

<!-- model-card:canonical:start -->
## Method overview

TimeBridge is a patch-based Transformer framework for multivariate long-term time-series forecasting that explicitly handles non-stationarity at two granularities: Integrated Attention removes short-term non-stationarity within each variate's patches to capture stable local dependencies, while Cointegrated Attention preserves non-stationarity across variates to model long-term cointegration relationships between channels.

## Core architecture

TimeBridge is a patch-based Transformer framework for multivariate long-term time-series forecasting that explicitly handles non-stationarity at two granularities: Integrated Attention removes short-term non-stationarity within each variate's patches to capture stable local dependencies, while Cointegrated Attention preserves non-stationarity across variates to model long-term cointegration relationships between channels.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2410.04442); title: TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting; venue/year: ICML 2025 / 2025
- [codebase](https://github.com/Hank0626/TimeBridge); revision: `0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TimeBridge.toml`](../../../../configs/models/TimeBridge.toml).

## Differences

**Paper-driven local implementation.** The implementation follows Equations
(3)–(10): moving-average detrending supplies stationary queries and keys for
intra-variate Integrated Attention while original patches remain values;
attention downsampling aggregates long context; Cointegrated Attention then
mixes variates without removing non-stationarity. Calendar marks are not part
of this local contract. The external repository is reference-only; no source
file was copied or adapted.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `period=24`, `num_p=2`, `ia_layers=2`, `pd_layers=1`, `ca_layers=2`, `stable_len=3`, `d_model=16`, `n_heads=4`, `d_ff=128`, `attn_dropout=0.15`, `dropout=0.0`, `activation='gelu'`, `revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting
- **Venue**: ICML 2025
- **Published**: 2025 (arXiv: 2024-10)
- **arXiv**: https://arxiv.org/abs/2410.04442

## Abstract
Non-stationarity poses significant challenges for multivariate time series forecasting due to the inherent short-term fluctuations and long-term trends that can lead to spurious regressions or obscure essential long-term relationships. Most existing methods either eliminate or retain non-stationarity without adequately addressing its distinct impacts on short-term and long-term modeling. Eliminating non-stationarity is essential for avoiding spurious regressions and capturing local dependencies in short-term modeling, while preserving it is crucial for revealing long-term cointegration across variates. In this paper, we propose TimeBridge, a novel framework designed to bridge the gap between non-stationarity and dependency modeling in long-term time series forecasting. By segmenting input series into smaller patches, TimeBridge applies Integrated Attention to mitigate short-term non-stationarity and capture stable dependencies within each variate, while Cointegrated Attention preserves non-stationarity to model long-term cointegration across variates. Extensive experiments show that TimeBridge consistently achieves state-of-the-art performance in both short-term and long-term forecasting. Additionally, TimeBridge demonstrates exceptional performance in financial forecasting on the CSI 500 and S&P 500 indices, further validating its robustness and effectiveness. Code is available at https://github.com/Hank0626/TimeBridge.

## In ModernTSF
Default config: `configs/models/TimeBridge.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

**Paper-driven local implementation.** The implementation follows Equations
(3)–(10): moving-average detrending supplies stationary queries and keys for
intra-variate Integrated Attention while original patches remain values;
attention downsampling aggregates long context; Cointegrated Attention then
mixes variates without removing non-stationarity. Calendar marks are not part
of this local contract. The external repository is reference-only; no source
file was copied or adapted.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/LiuWHL0BX25,
  author       = {Peiyuan Liu and
                  Beiliang Wu and
                  Yifan Hu and
                  Naiqi Li and
                  Tao Dai and
                  Jigang Bao and
                  Shu{-}Tao Xia},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/liu25cb.html},
  timestamp    = {Thu, 26 Feb 2026 08:16:33 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/LiuWHL0BX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
