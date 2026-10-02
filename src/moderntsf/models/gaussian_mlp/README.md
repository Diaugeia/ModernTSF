---
name: "GaussianMLP"
summary: "GaussianMLP is a simple **parametric probabilistic** baseline: an MLP maps the flattened input window to per-step Gaussian parameters `(loc, scale)` for every horizon step and channel, returning `(B, pred_len, C, 2)` with a strictly positive scale (`softplus + eps`). It is trained by maximum likelihood (`nll_gaussian`) and scored with the closed-form Gaussian CRPS plus coverage / width. It serves as the minimal reference for the `distribution` output type — the parametric counterpart to the quantile models."
paper: ""
paper_title: "Gaussian-head MLP (ModernTSF parametric probabilistic baseline)"
venue: "ModernTSF"
year: 2026
---
# GaussianMLP

GaussianMLP is a simple **parametric probabilistic** baseline: an MLP maps the
flattened input window to per-step Gaussian parameters `(loc, scale)` for every
horizon step and channel, returning `(B, pred_len, C, 2)` with a strictly
positive scale (`softplus + eps`). It is trained by maximum likelihood
(`nll_gaussian`) and scored with the closed-form Gaussian CRPS plus
coverage / width. It serves as the minimal reference for the `distribution`
output type — the parametric counterpart to the quantile models.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, parameters]` distribution parameters.

## Paper and code

- paper: not available; title: Gaussian-head MLP (ModernTSF parametric probabilistic baseline); venue/year: ModernTSF / 2026
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GaussianMLP.toml`](../../../../configs/models/GaussianMLP.toml).

## Differences

- Local implementation. This is an intentional in-repository baseline, not an external paper reproduction. Its defining map is `h_0 = vec(X)`, `h_l = Dropout(ReLU(W_l h_{l-1}+b_l))`, `loc = W_mu h`, and `scale = softplus(W_sigma h)+eps`.
- It predicts independent Gaussian location/scale pairs; cross-channel and cross-horizon covariance are not modeled.

## Shared components

- [`gaussian_parameter_head`](../_components/gaussian_parameter_head/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `hidden_size=256`, `num_layers=2`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Method
A standard parametric forecasting baseline (Gaussian likelihood head, as
popularized by DeepAR-style models). No single canonical paper; this is a
ModernTSF reference implementation of the `distribution` output axis.

## In ModernTSF
`output_type = "distribution"`, `distribution_family = "gaussian"`; pair with
`[training] loss = "nll_gaussian"`. Default config:
`configs/models/GaussianMLP.toml`; specification: `spec.py`; implementation:
`model.py`. See the `deepar` model for an RNN-based distribution forecaster.

## Source and verification

- Local implementation. This is an intentional in-repository baseline, not an external paper reproduction. Its defining map is `h_0 = vec(X)`, `h_l = Dropout(ReLU(W_l h_{l-1}+b_l))`, `loc = W_mu h`, and `scale = softplus(W_sigma h)+eps`.
- It predicts independent Gaussian location/scale pairs; cross-channel and cross-horizon covariance are not modeled.
