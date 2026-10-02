---
name: "MTLinear"
summary: "MTLinear treats multivariate forecasting as multi-task learning: variates are grouped a priori by correlation, each group owns one linear-family forecaster (DLinear, NLinear, Linear, or RLinear) shared by its variates, and training weights every horizon-by-variate squared error by a detached inverse error magnitude to balance gradients inside each group."
paper: "https://arxiv.org/abs/2502.03571"
paper_title: "A Multi-Task Learning Approach to Linear Multivariate Forecasting"
venue: "AISTATS 2025"
year: 2025
code: "https://github.com/azencot-group/MTLinear"
revision: "7d0d9fb71d0e3652829ed28dda5acc558b27f528"
license: "NOASSERTION"
tagline: "Correlation-clustered variate groups, one shared linear forecaster per group, trained with error-magnitude loss weights."
tags: ["linear", "decomposition", "channel-grouping", "multi-task", "gradient-balancing", "lightweight"]
composition: ["normalization=none", "decomposition=component:dlinear", "temporal=component:channel_wise_linear", "channel=local:correlation-grouped-heads", "head=component:channel_wise_linear", "loss=local:magnitude-penalized-squared-error"]
---
# MTLinear

## Key ideas

- `Model.fit_clusters` groups variates by complete-linkage agglomeration on `1 - corr` of the training series; the trainer calls it once through `ModelSpec.training_setup`, and the grouping is stored in the `group_of` buffer.
- `Model.forward` routes each group to its own `GroupHead` (`dlinear` backbone, or `channel_wise_linear` with last-value centering or `revin` for NLinear and RLinear), so variates in a group share weights and groups do not.
- `Model.penalty_weights` and `penalized_loss` implement the gradient magnitude penalty `w_ij = (K_j H_i)^-a` as constants per batch; `ModelSpec.training_objective` replaces the configured loss with it.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2502.03571); title: A Multi-Task Learning Approach to Linear Multivariate Forecasting; venue/year: AISTATS 2025 / 2025
- [codebase](https://github.com/azencot-group/MTLinear); revision: `7d0d9fb71d0e3652829ed28dda5acc558b27f528`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MTLinear.toml`](../../../../configs/models/MTLinear.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`dlinear`](../_components/dlinear/README.md)
- [`last_value_center`](../_components/last_value_center/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `layer_type='DLinear'`, `kernel_size=25`, `cluster_dist=0.5`, `max_clusters=8`, `penalty_param=2.0`, `use_horizon_penalty=True`, `use_variates_penalty=True`
<!-- model-card:canonical:end -->
