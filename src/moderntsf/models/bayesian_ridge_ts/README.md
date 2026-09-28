---
name: "BayesianRidgeTS"
summary: "BayesianRidgeTS is a channel-wise lag regression baseline with a learned Gaussian weight-prior precision, optimized as a differentiable MAP adaptation."
paper: "https://doi.org/10.1162/neco.1992.4.3.415"
paper_title: "Bayesian Interpolation"
venue: "Neural Computation"
year: 1992
---
# BayesianRidgeTS

BayesianRidgeTS is a channel-wise lag regression baseline with a learned Gaussian weight-prior precision, optimized as a differentiable MAP adaptation.

<!-- model-card:canonical:start -->
## Method overview

BayesianRidgeTS is a channel-wise lag regression baseline with a learned Gaussian weight-prior precision, optimized as a differentiable MAP adaptation.

## Core architecture

BayesianRidgeTS is a channel-wise lag regression baseline with a learned Gaussian weight-prior precision, optimized as a differentiable MAP adaptation.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1162/neco.1992.4.3.415); title: Bayesian Interpolation; venue/year: Neural Computation / 1992
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/BayesianRidgeTS.toml`](../../../../configs/models/BayesianRidgeTS.toml).

## Differences

This clean-room implementation uses the cited Bayesian linear-regression prior as its mathematical basis. It performs gradient-trained MAP forecasting; it does not implement MacKay evidence maximization, infer observation precision, or return posterior predictive uncertainty. No third-party implementation was inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `initial_weight_precision=0.001`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Bayesian Interpolation
- **Venue**: Neural Computation
- **Published**: 1992
- **Link**: https://doi.org/10.1162/neco.1992.4.3.415

## Abstract
Bayesian ridge regression places a Gaussian prior over linear-regression weights. The local MAP adaptation uses a learned positive prior precision and a shared channel-wise lag projection; it does not claim full posterior inference or uncertainty calibration.

## In ModernTSF
Default config: `configs/models/BayesianRidgeTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This clean-room implementation uses the cited Bayesian linear-regression prior as its mathematical basis. It performs gradient-trained MAP forecasting; it does not implement MacKay evidence maximization, infer observation precision, or return posterior predictive uncertainty. No third-party implementation was inspected or copied.

## Citation

```bibtex
@article{mackay1992bayesian,
  author  = {David J. C. MacKay},
  title   = {Bayesian Interpolation},
  journal = {Neural Computation},
  volume  = {4},
  number  = {3},
  pages   = {415--447},
  year    = {1992},
  doi     = {10.1162/neco.1992.4.3.415},
  url     = {https://doi.org/10.1162/neco.1992.4.3.415}
}
```
