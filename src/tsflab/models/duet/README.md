---
name: "DUET"
summary: "DUET (Dual Clustering Enhanced Multivariate Time Series Forecasting) is a time series forecasting model that addresses two key challenges in multivariate forecasting: heterogeneous temporal patterns caused by distribution shifts, and complex inter-channel correlations. It introduces a Temporal Clustering Module (TCM) that groups time segments into fine-grained distribution clusters and assigns specialised pattern extractors to each, and a Channel Clustering Module (CCM) that performs soft channel clustering in the frequency domain via metric learning and sparsification, jointly modelling both temporal and channel dimensions."
paper: "https://arxiv.org/abs/2412.10859"
paper_title: "DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting"
venue: "KDD 2025"
year: 2025
code: "https://github.com/decisionintelligence/DUET"
revision: "dcc6e6780a9138731b64b9b5398a94a1d97033f0"
license: "MIT"
---
# DUET

DUET (Dual Clustering Enhanced Multivariate Time Series Forecasting) is a time series forecasting model that addresses two key challenges in multivariate forecasting: heterogeneous temporal patterns caused by distribution shifts, and complex inter-channel correlations. It introduces a Temporal Clustering Module (TCM) that groups time segments into fine-grained distribution clusters and assigns specialised pattern extractors to each, and a Channel Clustering Module (CCM) that performs soft channel clustering in the frequency domain via metric learning and sparsification, jointly modelling both temporal and channel dimensions.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2412.10859); title: DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting; venue/year: KDD 2025 / 2025
- [codebase](https://github.com/decisionintelligence/DUET); revision: `dcc6e6780a9138731b64b9b5398a94a1d97033f0`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DUET.toml`](../../../../configs/models/DUET.toml).

## Differences

Clean-room implementation: confirmed.

Clean-room structure map: distributional router; trend/seasonal temporal experts; top-k mixture with trainable residual; Mahalanobis attention bias; channel encoder; direct horizon head. No source code or checkpoint is reused.

The gating MLP and the top-k sparsification with floor-renormalization were
extracted into the shared `topk_expert_router` component; the distributional
(mean/std) feature construction, Mahalanobis attention bias, and channel
encoder remain model-local paper-specific logic.

The per-expert moving average stays model-local rather than routed through
`series_decomposition.EdgePaddedMovingAverage`: that component validates
`kernel_size` as strictly odd, and BiST relies on that validation to reject
misconfigured even kernel sizes. DUET's local helper additionally accepts
even kernels via asymmetric padding, so extending the shared component to
match it would silently change BiST's error contract.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`topk_expert_router`](../_components/topk_expert_router/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `n_heads=4`, `e_layers=2`, `d_ff=64`, `dropout=0.1`, `fc_dropout=0.1`, `moving_avg=25`, `num_experts=4`, `k=2`, `hidden_size=64`, `noisy_gating=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting
- **Venue**: KDD 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2412.10859

## Abstract
Multivariate time series forecasting is crucial for various applications, such as financial investment, energy management, weather forecasting, and traffic optimization. However, accurate forecasting is challenging due to two main factors. First, real-world time series often show heterogeneous temporal patterns caused by distribution shifts over time. Second, correlations among channels are complex and intertwined, making it hard to model the interactions among channels precisely and flexibly. In this study, we address these challenges by proposing a general framework called DUET, which introduces dual clustering on the temporal and channel dimensions to enhance multivariate time series forecasting. First, we design a Temporal Clustering Module (TCM) that clusters time series into fine-grained distributions to handle heterogeneous temporal patterns. For different distribution clusters, we design various pattern extractors to capture their intrinsic temporal patterns, thus modeling the heterogeneity. Second, we introduce a novel Channel-Soft-Clustering strategy and design a Channel Clustering Module (CCM), which captures the relationships among channels in the frequency domain through metric learning and applies sparsification to mitigate the adverse effects of noisy channels. Finally, DUET combines TCM and CCM to incorporate both the temporal and channel dimensions. Extensive experiments on 25 real-world datasets from 10 application domains, demonstrate the state-of-the-art performance of DUET.

## In TSFLab
Default config: `configs/models/DUET.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed.

Clean-room structure map: distributional router; trend/seasonal temporal experts; top-k mixture with trainable residual; Mahalanobis attention bias; channel encoder; direct horizon head. No source code or checkpoint is reused.

The gating MLP and the top-k sparsification with floor-renormalization were
extracted into the shared `topk_expert_router` component; the distributional
(mean/std) feature construction, Mahalanobis attention bias, and channel
encoder remain model-local paper-specific logic.

The per-expert moving average stays model-local rather than routed through
`series_decomposition.EdgePaddedMovingAverage`: that component validates
`kernel_size` as strictly odd, and BiST relies on that validation to reject
misconfigured even kernel sizes. DUET's local helper additionally accepts
even kernels via asymmetric padding, so extending the shared component to
match it would silently change BiST's error contract.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/QiuW0GH025,
  author       = {Xiangfei Qiu and
                  Xingjian Wu and
                  Yan Lin and
                  Chenjuan Guo and
                  Jilin Hu and
                  Bin Yang},
  editor       = {Yizhou Sun and
                  Flavio Chierichetti and
                  Hady W. Lauw and
                  Claudia Perlich and
                  Wee Hyong Tok and
                  Andrew Tomkins},
  title        = {{DUET:} Dual Clustering Enhanced Multivariate Time Series Forecasting},
  booktitle    = {Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, V.1, {KDD} 2025, Toronto, ON, Canada, August 3-7,
                  2025},
  pages        = {1185--1196},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3690624.3709325},
  doi          = {10.1145/3690624.3709325},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/QiuW0GH025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
