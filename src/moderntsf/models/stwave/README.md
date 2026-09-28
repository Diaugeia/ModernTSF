---
name: "STWave"
summary: "STWave is a spatiotemporal forecasting model for traffic flow prediction that disentangles non-stationary traffic sequences into long-term (low-frequency) trend components and short-term (high-frequency) event components using discrete wavelet transform. A dual-channel encoder processes each frequency band separately with an efficient spectral graph attention mechanism that incorporates wavelet-based graph positional encoding and a query sampling strategy to reduce the quadratic complexity of full graph attention while preserving spatial expressiveness."
paper: "https://arxiv.org/abs/2112.02740"
paper_title: "When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks"
venue: "ICDE 2023"
year: 2023
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# STWave

STWave is a spatiotemporal forecasting model for traffic flow prediction that disentangles non-stationary traffic sequences into long-term (low-frequency) trend components and short-term (high-frequency) event components using discrete wavelet transform. A dual-channel encoder processes each frequency band separately with an efficient spectral graph attention mechanism that incorporates wavelet-based graph positional encoding and a query sampling strategy to reduce the quadratic complexity of full graph attention while preserving spatial expressiveness.

<!-- model-card:canonical:start -->
## Method overview

STWave is a spatiotemporal forecasting model for traffic flow prediction that disentangles non-stationary traffic sequences into long-term (low-frequency) trend components and short-term (high-frequency) event components using discrete wavelet transform.

## Core architecture

A dual-channel encoder processes each frequency band separately with an efficient spectral graph attention mechanism that incorporates wavelet-based graph positional encoding and a query sampling strategy to reduce the quadratic complexity of full graph attention while preserving spatial expressiveness.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2112.02740); title: When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks; venue/year: ICDE 2023 / 2023
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/STWave.toml`](../../../../configs/models/STWave.toml).

## Differences

Clean-room implementation: confirmed. The reference-only source code was not copied.

The implementation is independently derived from the paper and contains no
BasicTS source. It retains temporal wavelet disentanglement, dual encoders,
spectral graph positions, query sampling, and adaptive low/high fusion. It uses
a Haar basis and omits the paper's auxiliary low-frequency loss and published
dataset preprocessing.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `hidden_size=6`, `layers=1`, `log_samples=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks
- **Venue**: ICDE 2023
- **Published**: 2023 (arXiv: 2021-12)
- **arXiv**: https://arxiv.org/abs/2112.02740

## Abstract
Traffic forecasting is crucial for public safety and resource optimization, yet is very challenging due to three aspects: i) current existing works mostly exploit intricate temporal patterns (e.g., the short-term thunderstorm and long-term daily trends) within a single method, which fail to accurately capture spatio-temporal dependencies under different schemas; ii) the under-exploration of the graph positional encoding limit the extraction of spatial information in the commonly used full graph attention network; iii) the quadratic complexity of the full graph attention introduces heavy computational needs. To achieve the effective traffic flow forecasting, we propose an efficient spectral graph attention network with disentangled traffic sequences. Specifically, the discrete wavelet transform is leveraged to obtain the low- and high-frequency components of traffic sequences, and a dual-channel encoder is elaborately designed to accurately capture the spatio-temporal dependencies under long- and short-term schemas of the low- and high-frequency components. Moreover, a novel wavelet-based graph positional encoding and a query sampling strategy are introduced in our spectral graph attention to effectively guide message passing and efficiently calculate the attention. Extensive experiments on four real-world datasets show the superiority of our model, i.e., the higher traffic forecasting precision with lower computational cost.

## In ModernTSF
Default config: `configs/models/STWave.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Clean-room implementation: confirmed. The reference-only source code was not copied.

The implementation is independently derived from the paper and contains no
BasicTS source. It retains temporal wavelet disentanglement, dual encoders,
spectral graph positions, query sampling, and adaptive low/high fusion. It uses
a Haar basis and omits the paper's auxiliary low-frequency loss and published
dataset preprocessing.

## Citation

```bibtex
@inproceedings{DBLP:conf/icde/FangQL0XZ023,
  author       = {Yuchen Fang and
                  Yanjun Qin and
                  Haiyong Luo and
                  Fang Zhao and
                  Bingbing Xu and
                  Liang Zeng and
                  Chenxing Wang},
  title        = {When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting
                  via Efficient Spectral Graph Attention Networks},
  booktitle    = {39th {IEEE} International Conference on Data Engineering, {ICDE} 2023,
                  Anaheim, CA, USA, April 3-7, 2023},
  pages        = {517--529},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/ICDE55515.2023.00046},
  doi          = {10.1109/ICDE55515.2023.00046},
  timestamp    = {Sun, 02 Nov 2025 21:27:15 +0100},
  biburl       = {https://dblp.org/rec/conf/icde/FangQL0XZ023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
