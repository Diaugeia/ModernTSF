---
name: "RidgeRegressionTS"
summary: "RidgeRegressionTS applies a shared channel-wise lag projection to the forecast horizon and exposes the ridge L2 weight penalty through `aux_loss` for the standard trainer."
paper: "https://doi.org/10.1080/00401706.1970.10488634"
paper_title: "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
venue: "Technometrics 1970"
year: 1970
tagline: "One shared linear lag-to-horizon map per channel with an explicit L2 weight penalty returned as aux_loss."
tags: ["linear", "statistical", "regularization", "channel-independent", "baseline", "lightweight"]
composition: ["normalization=none", "decomposition=none", "temporal=local:direct-linear-lag-regression", "channel=local:channel-independent-shared-weights", "head=local:linear-horizon-projection", "loss=loss:mse+local:l2-weight-penalty-aux-loss"]
---
# RidgeRegressionTS

## Key ideas

- A single `nn.Linear(seq_len, pred_len)` is applied over time and shared by all channels, with no normalization.
- `forward` sets `aux_loss = l2_penalty * ||W||^2` (default 1e-4), which the standard trainer adds to the task loss, making the fit a gradient-trained ridge regression.
- Intended as a regularized linear baseline.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1080/00401706.1970.10488634); title: Ridge Regression: Biased Estimation for Nonorthogonal Problems; venue/year: Technometrics 1970 / 1970
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/RidgeRegressionTS.toml`](../../../../configs/models/RidgeRegressionTS.toml).

## Differences

This is an independent implementation from the cited ridge objective; no
external source implementation was inspected or copied. It optimizes a direct
multi-horizon lag projection with gradient descent rather than solving the
closed-form ridge estimator. Coefficients are shared across channels, and the L2
weight term is exposed to the trainer as `aux_loss`.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `l2_penalty=0.0001`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Ridge regression is a classical regularized linear model that extends ordinary least squares by adding an L2 penalty on the regression coefficients (Tikhonov regularization). Applied to time series forecasting, the model treats lagged values of all channels as input features and predicts the future horizon via a single linear layer whose weights are regularized to avoid overfitting. The L2 penalty shrinks large coefficients toward zero, improving generalization on high-dimensional or correlated feature sets. In the TSFLab context, the model is implemented as a `torch.nn.Module` with a learnable linear layer and a configurable regularization strength, enabling GPU-accelerated training through the standard benchmark trainer alongside all other model classes.

## In TSFLab
Default config: `configs/models/RidgeRegressionTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent implementation from the cited ridge objective; no
external source implementation was inspected or copied. It optimizes a direct
multi-horizon lag projection with gradient descent rather than solving the
closed-form ridge estimator. Coefficients are shared across channels, and the L2
weight term is exposed to the trainer as `aux_loss`.

## Citation

```bibtex
@article{hoerl1970ridge,
  author  = {Arthur E. Hoerl and Robert W. Kennard},
  title   = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  volume  = {12},
  number  = {1},
  pages   = {55--67},
  year    = {1970},
  doi     = {10.1080/00401706.1970.10488634}
}
```
