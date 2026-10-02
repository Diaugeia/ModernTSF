---
name: "DistDF"
summary: "DistDF is a clean-room joint-distribution Bures-Wasserstein training objective paired with a compact channel-wise direct forecaster for the common runtime interface."
paper: "https://arxiv.org/abs/2510.24574"
paper_title: "DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment"
venue: "ICLR 2026"
year: 2026
code: "https://github.com/Master-PLC/DistDF"
revision: "21b050fc230d35c7e1c4507c8da3dcd81dc9e1b9"
license: "MIT"
tagline: "Training objective matching joint history-forecast Gaussian moments (Bures-Wasserstein) on a compact linear forecaster."
tags: ["linear", "loss-framework", "distribution-alignment", "normalization", "channel-independent", "lightweight"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:channel_wise_linear", "channel=local:channel-independent-shared-weights", "head=local:linear-forecast", "loss=loss:mse+local:joint-bures-wasserstein-distribution-loss"]
---
# DistDF

## Key ideas

- DistDF changes the loss, not the architecture: the forecaster is a shared `channel_wise_linear` map wrapped by `revin`.
- `joint_distribution_discrepancy` forms joint samples [history, target] and [history, forecast] over batch-channel pairs, estimates Gaussian mean and covariance (with `covariance_eps` jitter), and computes the squared Bures-Wasserstein distance.
- `training_objective` combines it with MSE as gamma * discrepancy + (1 - gamma) * MSE; plain `forward` is point forecasting.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.24574); title: DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/Master-PLC/DistDF); revision: `21b050fc230d35c7e1c4507c8da3dcd81dc9e1b9`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DistDF.toml`](../../../../configs/models/DistDF.toml).

## Differences

Clean-room implementation: confirmed. The reference-only artifact was not
inspected or copied. The rewrite follows Algorithm 1 and equations (5)--(6): it
forms `[X,Y]` and `[X,Yhat]`, estimates joint Gaussian moments, evaluates the
Bures-Wasserstein expression, and combines it with MSE.

The paper uses several external backbones; this entry supplies a compact shared
linear carrier. Batch-channel pairs form empirical samples and positive jitter
stabilizes small covariances. Experiments must call `training_loss` to activate
DistDF. Evidence is in `../../../../verification/evidence/DistDF.json`.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `gamma=0.1`, `covariance_eps=1e-05`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.24574

## Abstract
Training time-series forecasting models requires aligning the conditional distribution of model forecasts with that of the label sequence. The standard direct forecast (DF) approach resorts to minimizing the conditional negative log-likelihood, typically estimated by the mean squared error. However, this estimation proves biased when the label sequence exhibits autocorrelation. In this paper, we propose DistDF, which achieves alignment by minimizing a distributional discrepancy between the conditional distributions of forecast and label sequences. Since such conditional discrepancies are difficult to estimate from finite time-series observations, we introduce a joint-distribution Wasserstein discrepancy for time-series forecasting, which provably upper bounds the conditional discrepancy of interest. The proposed discrepancy is tractable, differentiable, and readily compatible with gradient-based optimization. Extensive experiments show that DistDF improves diverse forecasting models and achieves leading performance.

## Source and verification

Clean-room implementation: confirmed. The reference-only artifact was not
inspected or copied. The rewrite follows Algorithm 1 and equations (5)--(6): it
forms `[X,Y]` and `[X,Yhat]`, estimates joint Gaussian moments, evaluates the
Bures-Wasserstein expression, and combines it with MSE.

The paper uses several external backbones; this entry supplies a compact shared
linear carrier. Batch-channel pairs form empirical samples and positive jitter
stabilizes small covariances. Experiments must call `training_loss` to activate
DistDF. Evidence is in `../../../../verification/evidence/DistDF.json`.

## In TSFLab
Default config: `configs/models/DistDF.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@misc{wang2025distdf,
  author        = {Hao Wang and
                  Licheng Pan and
                  Yuan Lu and
                  Zhixuan Chu and
                  Xiaoxi Li and
                  Shuting He and
                  Zhichao Chen and
                  Haoxuan Li and
                  Qingsong Wen and
                  Zhouchen Lin},
  title         = {DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment},
  year          = {2025},
  eprint        = {2510.24574},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.24574}
}
```
