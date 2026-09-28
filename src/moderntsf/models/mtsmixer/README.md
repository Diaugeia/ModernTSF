---
name: "MTSMixer"
summary: "MTSMixer is an MLP-Mixer-based model for multivariate time-series forecasting that replaces Transformer attention with two factorised mixing modules: one captures temporal dependencies and another captures cross-channel dependencies, avoiding the entanglement and redundancy introduced by joint attention. It also explicitly models the input-to-prediction mapping, yielding strong accuracy with significantly lower computational cost than Transformer-based baselines."
paper: "https://arxiv.org/abs/2302.04501"
paper_title: "MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing"
venue: "IJCNN 2025"
year: 2025
code: "https://github.com/plumprc/MTS-Mixers"
revision: "262448f00cf8b7e0ee38ef2ca510cc70ed4b8dc8"
license: "NOASSERTION"
---
# MTSMixer

MTSMixer is an MLP-Mixer-based model for multivariate time-series forecasting that replaces Transformer attention with two factorised mixing modules: one captures temporal dependencies and another captures cross-channel dependencies, avoiding the entanglement and redundancy introduced by joint attention. It also explicitly models the input-to-prediction mapping, yielding strong accuracy with significantly lower computational cost than Transformer-based baselines.

<!-- model-card:canonical:start -->
## Method overview

MTSMixer is an MLP-Mixer-based model for multivariate time-series forecasting that replaces Transformer attention with two factorised mixing modules: one captures temporal dependencies and another captures cross-channel dependencies, avoiding the entanglement and redundancy introduced by joint attention.

## Core architecture

It also explicitly models the input-to-prediction mapping, yielding strong accuracy with significantly lower computational cost than Transformer-based baselines.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2302.04501); title: MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing; venue/year: IJCNN 2025 / 2025
- [codebase](https://github.com/plumprc/MTS-Mixers); revision: `262448f00cf8b7e0ee38ef2ca510cc70ed4b8dc8`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/MTSMixer.toml`](../../../../configs/models/MTSMixer.toml).

## Differences

Pinned source inspection: `models/MTSMixer.py` was examined at the recorded
revision to confirm implementation details. The local module was written for
ModernTSF; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from paper equations (3), (6), and (8); the unlicensed reference repository was inspected at the pinned revision; no external source code was copied. The default uses equidistant interleaved temporal subsequences, independent temporal MLPs, a low-rank channel bottleneck, residual composition, RevIN, and a direct history-to-horizon projection. Attention/random-matrix variants and SVD/NMF refinement are omitted; GELU, pre-LayerNorm, and the compact forecast-only runtime are disclosed local choices rather than benchmark-reference comparison claims.

Component audit (`mixer_block`): `FactorizedMixerBlock` normalizes over the channel axis only (not the joint `(seq_len, channels)` shape), mixes through the subsequence-grouped `TemporalSubsequenceMixer` (Eq. 6) instead of a single dense time projection, and uses a low-rank `ChannelInteraction` bottleneck (Eq. 8) instead of the hidden-width feature MLP; it is not the same operator and stays model-local.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `d_ff=4`, `e_layers=2`, `fac_T=True`, `fac_C=True`, `sampling=2`, `norm=True`, `individual=False`, `rev=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing
- **Venue**: IJCNN 2025
- **Published**: 2025 (arXiv: 2023-02)
- **arXiv**: https://arxiv.org/abs/2302.04501

## Abstract
Multivariate time series forecasting has been widely used in various practical scenarios. Recently, Transformer-based models have shown significant potential in forecasting tasks due to the capture of long-range dependencies. However, recent studies in the vision and NLP fields show that the role of attention modules is not clear, which can be replaced by other token aggregation operations. This paper investigates the contributions and deficiencies of attention mechanisms on the performance of time series forecasting. Specifically, we find that (1) attention is not necessary for capturing temporal dependencies, (2) the entanglement and redundancy in the capture of temporal and channel interaction affect the forecasting performance, and (3) it is important to model the mapping between the input and the prediction sequence. To this end, we propose MTS-Mixers, which use two factorized modules to capture temporal and channel dependencies. Experimental results on several real-world datasets show that MTS-Mixers outperform existing Transformer-based models with higher efficiency.

## In ModernTSF
Default config: `configs/models/MTSMixer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Pinned source inspection: `models/MTSMixer.py` was examined at the recorded
revision to confirm implementation details. The local module was written for
ModernTSF; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from paper equations (3), (6), and (8); the unlicensed reference repository was inspected at the pinned revision; no external source code was copied. The default uses equidistant interleaved temporal subsequences, independent temporal MLPs, a low-rank channel bottleneck, residual composition, RevIN, and a direct history-to-horizon projection. Attention/random-matrix variants and SVD/NMF refinement are omitted; GELU, pre-LayerNorm, and the compact forecast-only runtime are disclosed local choices rather than benchmark-reference comparison claims.

Component audit (`mixer_block`): `FactorizedMixerBlock` normalizes over the channel axis only (not the joint `(seq_len, channels)` shape), mixes through the subsequence-grouped `TemporalSubsequenceMixer` (Eq. 6) instead of a single dense time projection, and uses a low-rank `ChannelInteraction` bottleneck (Eq. 8) instead of the hidden-width feature MLP; it is not the same operator and stays model-local.

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcnn/LiLRPX25,
  author       = {Zhe Li and
                  Xuanxuan Li and
                  Zhongwen Rao and
                  Lujia Pan and
                  Zenglin Xu},
  title        = {MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal
                  and Channel Mixing},
  booktitle    = {International Joint Conference on Neural Networks, {IJCNN} 2025, Rome,
                  Italy, June 30 - July 5, 2025},
  pages        = {1--8},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/IJCNN64981.2025.11229402},
  doi          = {10.1109/IJCNN64981.2025.11229402},
  timestamp    = {Fri, 21 Nov 2025 20:23:55 +0100},
  biburl       = {https://dblp.org/rec/conf/ijcnn/LiLRPX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
