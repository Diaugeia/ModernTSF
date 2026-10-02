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
- `Model.penalty_weights` and `penalized_loss` implement the gradient magnitude penalty `w_ij = (K_j H_i)^-a` as constants per batch; `ModelSpec.training_objective` replaces the configured loss with it whenever the horizon or variate penalty is on (with both off, the configured criterion is used).

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

- Implementation: independent local rewrite from Sections 4.2 to 4.4 (Eqs. 7 and 8). The official repository (`azencot-group/MTLinear`, revision `7d0d9fb71d0e3652829ed28dda5acc558b27f528`) has no LICENSE file (the license field records that absence), so it was read only to resolve omissions (`models/MTLinear.py`, `utils/grad_manip.py`, `exp/exp_main.py`, `layers/Linear_layers.py`); nothing was copied.
- Resolved from the official code: complete-linkage agglomeration on `1 - corr`, merging only below the threshold; the penalty uses the batch mean absolute error per group (`H_i` over horizon and batch, `K_j` over variates and batch), zero errors are replaced by 1 before inversion, the weights are detached constants, the horizon-only penalty is scaled by `group size / enc_in`, and the variate-only penalty is broadcast over the horizon; DLinear/NLinear/Linear/RLinear heads are shared across the variates of a group.
- Differences from the official code: (1) `cluster_dist` is always a distance threshold; the official code treats an integer value as a fixed cluster count, and has no `max_clusters` cap (here the grouping is re-cut to at most `max_clusters` groups). (2) The grouping is fitted once through `ModelSpec.training_setup` from the newest step of every training window, not at model construction from the full scaled training series; before it runs, variates are assigned round-robin, and NaN correlations (constant series) are set to 0. (3) The official multi-block early stopping (`multi_early_stopping`, one best checkpoint per group) is not implemented; the runner's single early-stopping rule applies. (4) With `MS` features the weights use the trailing channels' groups. (5) `kernel_size` is a parameter (official 25); per-dataset settings (for example sequence length 36 and learning rate 0.01 for the illness grid) are not preset. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: clustering is checked against a complete-linkage reference in `tests/test_mtlinear.py`; no training was run.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`dlinear`](../_components/dlinear/README.md)
- [`last_value_center`](../_components/last_value_center/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `layer_type='DLinear'`, `kernel_size=25`, `cluster_dist=0.5`, `max_clusters=8`, `penalty_param=2.0`, `use_horizon_penalty=True`, `use_variates_penalty=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A Multi-Task Learning Approach to Linear Multivariate Forecasting
- **Venue**: AISTATS 2025
- **Published**: 2025 (arXiv: 2025-02)
- **arXiv**: https://arxiv.org/abs/2502.03571

## Source and verification

- Implementation: independent local rewrite from Sections 4.2 to 4.4 (Eqs. 7 and 8). The official repository (`azencot-group/MTLinear`, revision `7d0d9fb71d0e3652829ed28dda5acc558b27f528`) has no LICENSE file (the license field records that absence), so it was read only to resolve omissions (`models/MTLinear.py`, `utils/grad_manip.py`, `exp/exp_main.py`, `layers/Linear_layers.py`); nothing was copied.
- Resolved from the official code: complete-linkage agglomeration on `1 - corr`, merging only below the threshold; the penalty uses the batch mean absolute error per group (`H_i` over horizon and batch, `K_j` over variates and batch), zero errors are replaced by 1 before inversion, the weights are detached constants, the horizon-only penalty is scaled by `group size / enc_in`, and the variate-only penalty is broadcast over the horizon; DLinear/NLinear/Linear/RLinear heads are shared across the variates of a group.
- Differences from the official code: (1) `cluster_dist` is always a distance threshold; the official code treats an integer value as a fixed cluster count, and has no `max_clusters` cap (here the grouping is re-cut to at most `max_clusters` groups). (2) The grouping is fitted once through `ModelSpec.training_setup` from the newest step of every training window, not at model construction from the full scaled training series; before it runs, variates are assigned round-robin, and NaN correlations (constant series) are set to 0. (3) The official multi-block early stopping (`multi_early_stopping`, one best checkpoint per group) is not implemented; the runner's single early-stopping rule applies. (4) With `MS` features the weights use the trailing channels' groups. (5) `kernel_size` is a parameter (official 25); per-dataset settings (for example sequence length 36 and learning rate 0.01 for the illness grid) are not preset. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: clustering is checked against a complete-linkage reference in `tests/test_mtlinear.py`; no training was run.

## Citation

```bibtex
@inproceedings{nochumsohn2025mtlinear,
  title         = {A Multi-Task Learning Approach to Linear Multivariate Forecasting},
  author        = {Nochumsohn, Liran and Zisling, Hedi and Azencot, Omri},
  booktitle     = {Proceedings of the 28th International Conference on Artificial Intelligence and Statistics},
  year          = {2025},
  eprint        = {2502.03571},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2502.03571}
}
```
