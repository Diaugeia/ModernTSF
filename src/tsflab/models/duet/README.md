---
name: "DUET"
description: "Distribution-routed mixture of trend/seasonal linear experts, then distance-biased attention across channels. Use for multivariate forecasting with heterogeneous temporal distributions and intertwined channel correlations; not for univariate series or very many channels (pairwise channel attention)."
---

# DUET

## Idea

- Temporal clustering: a gating MLP (`topk_expert_router`) reads each sample's per-channel mean and std, and `topk_dense_mix` keeps the top-k experts, routing samples by distribution.
- Each `TemporalExpert` splits the input with its own moving-average kernel (`moving_avg`, `moving_avg - 2`, ...) and maps trend and seasonal parts with separate linear layers.
- Channel clustering: `ChannelAttention` runs over channel tokens with a bias of negative pairwise distances between standardized channels (`mahalanobis_bias`).
- A linear head maps channel tokens to the horizon; `revin` wraps the model.

## When to use

- Designed for multivariate data whose temporal patterns are heterogeneous because the distribution shifts over time, and whose channel correlations are complex (paper: 25 datasets, 10 domains).
- The channel bias favours mixing similar channels; useful when groups of channels co-move.
- Pairwise channel attention costs grow quadratically with channel count.

## Configure

- `enc_in`: number of channels (the router reads `2 * enc_in` mean/std features).

Other hyperparameters: preset defaults in `configs/models/DUET.toml`; tune generically.

## Differences

Clean-room implementation after reading `decisionintelligence/DUET` at `dcc6e678` (MIT); no source code or checkpoint reused.

- Structure: distributional router, trend/seasonal experts, top-k mixture, channel bias, channel encoder, direct horizon head.
- The channel bias is a fixed negative distance between standardized time-domain series; the abstract's frequency-domain metric learning and sparsification of noisy channels are not represented separately.
- The gating MLP and top-k sparsification (floor renormalization) come from the shared `topk_expert_router`.
- The per-expert moving average stays local: it accepts even kernels by asymmetric padding, which the shared odd-only `series_decomposition` (relied on by BiST) does not.

Citation: Qiu, X., Wu, X., Lin, Y., Guo, C., Hu, J., Yang, B. "DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting." KDD 2025, pp. 1185-1196. doi:10.1145/3690624.3709325.
